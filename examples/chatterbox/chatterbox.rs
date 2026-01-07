use std::path::Path;
use std::f32::consts::PI;

use anyhow::Result;
use ndarray::{Array1, Array2, Array3, Array4, Axis, s};
use ndarray_rand::rand_distr::StandardNormal;
use ort::session::{Session, SessionInputValue, builder::GraphOptimizationLevel};
use rand::prelude::*;
use rustfft::{FftPlanner, num_complex::Complex};
use safetensors::SafeTensors;
use tokenizers::Tokenizer;
use tracing::{info, error};

// Include common code for `ort` examples that allows using the various feature flags to enable different EPs and
// backends.
#[path = "../common/mod.rs"]
mod common;

/// A state-of-the-art native Rust Chatterbox Turbo speaker example.
/// Demonstrates high-performance TTS using `ort` with multiple models.
pub struct ChatterboxTurbo {
    t3_session: Session,
    flow_encoder: Session,
    flow_estimator: Session,
    vocoder_core: Session,
    tokenizer: Tokenizer,
    // Weights
    text_emb: Array2<f32>,
    speech_emb: Array2<f32>,
    speech_head_weight: Array2<f32>,
    speech_head_bias: Array1<f32>,
    t3_cond_emb: Array3<f32>,
    s3_speaker_emb: Array2<f32>,
    pub sample_rate: u32,
}

const KV_NAMES: [[&str; 2]; 24] = [
    ["past_key_values.0.key", "past_key_values.0.value"],
    ["past_key_values.1.key", "past_key_values.1.value"],
    ["past_key_values.2.key", "past_key_values.2.value"],
    ["past_key_values.3.key", "past_key_values.3.value"],
    ["past_key_values.4.key", "past_key_values.4.value"],
    ["past_key_values.5.key", "past_key_values.5.value"],
    ["past_key_values.6.key", "past_key_values.6.value"],
    ["past_key_values.7.key", "past_key_values.7.value"],
    ["past_key_values.8.key", "past_key_values.8.value"],
    ["past_key_values.9.key", "past_key_values.9.value"],
    ["past_key_values.10.key", "past_key_values.10.value"],
    ["past_key_values.11.key", "past_key_values.11.value"],
    ["past_key_values.12.key", "past_key_values.12.value"],
    ["past_key_values.13.key", "past_key_values.13.value"],
    ["past_key_values.14.key", "past_key_values.14.value"],
    ["past_key_values.15.key", "past_key_values.15.value"],
    ["past_key_values.16.key", "past_key_values.16.value"],
    ["past_key_values.17.key", "past_key_values.17.value"],
    ["past_key_values.18.key", "past_key_values.18.value"],
    ["past_key_values.19.key", "past_key_values.19.value"],
    ["past_key_values.20.key", "past_key_values.20.value"],
    ["past_key_values.21.key", "past_key_values.21.value"],
    ["past_key_values.22.key", "past_key_values.22.value"],
    ["past_key_values.23.key", "past_key_values.23.value"],
];

impl ChatterboxTurbo {
    pub fn new(artifact_dir: &Path) -> Result<Self> {
        info!("ChatterboxTurbo: Loading native Rust engine...");

        let builder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?;

        #[cfg(feature = "mps")]
        {
            use ort::ep::ExecutionProvider;
            let mps = ort::ep::MPS::default();
            if mps.is_available()? {
                builder = builder.with_execution_providers([mps.build()])?;
                info!("ChatterboxTurbo: Using MPS Execution Provider");
            }
        }

        let t3_session = builder.clone().commit_from_file(artifact_dir.join("t3_turbo.onnx"))?;
        let flow_encoder = builder.clone().commit_from_file(artifact_dir.join("s3_flow_encoder.onnx"))?;
        let flow_estimator = builder.clone().commit_from_file(artifact_dir.join("s3_flow_estimator.onnx"))?;
        let vocoder_core = builder.commit_from_file(artifact_dir.join("s3_vocoder_core.onnx"))?;

        let tokenizer = Tokenizer::from_file(artifact_dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("Tokenizer error: {}", e))?;

        let weights_path = artifact_dir.join("speaker_weights.safetensors");
        let weights_data = std::fs::read(&weights_path)?;
        let tensors = SafeTensors::deserialize(&weights_data)?;

        let load_2d = |name: &str| -> Result<Array2<f32>> {
            let view = tensors.tensor(name)?;
            let shape = view.shape();
            let data = view.data();
            let f32_data: Vec<f32> = data.chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            Ok(Array2::from_shape_vec((shape[0] as usize, shape[1] as usize), f32_data)?)
        };
        
        let load_1d = |name: &str| -> Result<Array1<f32>> {
            let view = tensors.tensor(name)?;
            let data = view.data();
            let f32_data: Vec<f32> = data.chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            Ok(Array1::from_vec(f32_data))
        };

        let load_3d = |name: &str| -> Result<Array3<f32>> {
            let view = tensors.tensor(name)?;
            let shape = view.shape();
            let data = view.data();
            let f32_data: Vec<f32> = data.chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            Ok(Array3::from_shape_vec((shape[0] as usize, shape[1] as usize, shape[2] as usize), f32_data)?)
        };

        Ok(Self {
            t3_session,
            flow_encoder,
            flow_estimator,
            vocoder_core,
            tokenizer,
            text_emb: load_2d("text_emb.weight")?,
            speech_emb: load_2d("speech_emb.weight")?,
            speech_head_weight: load_2d("speech_head.weight")?,
            speech_head_bias: load_1d("speech_head.bias")?,
            t3_cond_emb: load_3d("t3_cond_emb")?,
            s3_speaker_emb: load_2d("s3_speaker_emb")?,
            sample_rate: 24000,
        })
    }

    pub fn generate(&mut self, text: &str) -> Result<Vec<f32>> {
        let tokens = self.generate_tokens(text)?;
        let mel = self.generate_mel(&tokens)?;
        self.generate_waveform(&mel)
    }

    fn generate_tokens(&mut self, text: &str) -> Result<Vec<i64>> {
        let encoding = self.tokenizer.encode(text, true).map_err(|e| anyhow::anyhow!(e))?;
        let text_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        
        let mut text_embeds = Array2::zeros((text_ids.len(), 1024));
        for (i, &id) in text_ids.iter().enumerate() {
            let row_idx = (id as usize).min(self.text_emb.shape()[0] - 1);
            text_embeds.row_mut(i).assign(&self.text_emb.row(row_idx));
        }

        let cond_view = self.t3_cond_emb.slice(s![0, .., ..]);
        let prefix_embeds = ndarray::concatenate(Axis(0), &[cond_view, text_embeds.view()])?;

        let mut speech_ids = vec![6561];
        let mut kv_cache: Option<Vec<ort::value::Value>> = None;

        let mut dummy_kvs = Vec::with_capacity(48);
        for _ in 0..48 {
            dummy_kvs.push(ort::value::Value::from_array(Array4::<f32>::zeros((1, 16, 0, 64)))?);
        }

        for i in 0..1024 {
            let input_tensor = if i == 0 {
                let mut speech_start_embeds = Array2::zeros((1, 1024));
                speech_start_embeds.row_mut(0).assign(&self.speech_emb.row(6561));
                ndarray::concatenate(Axis(0), &[prefix_embeds.view(), speech_start_embeds.view()])?.insert_axis(Axis(0))
            } else {
                let last_id = *speech_ids.last().unwrap();
                let mut last_embed = Array2::zeros((1, 1024));
                let row_idx = (last_id as usize).min(self.speech_emb.shape()[0] - 1);
                last_embed.row_mut(0).assign(&self.speech_emb.row(row_idx));
                last_embed.insert_axis(Axis(0))
            };

            let mut inputs: Vec<(String, SessionInputValue)> = vec![
                ("inputs_embeds".to_string(), SessionInputValue::from(ort::value::Value::from_array(input_tensor.to_owned())?))
            ];
            
            if let Some(ref kvs) = kv_cache {
                for (j, kv) in kvs.iter().enumerate() {
                    let layer_idx = j / 2;
                    let name = if j % 2 == 0 { KV_NAMES[layer_idx][0] } else { KV_NAMES[layer_idx][1] };
                    inputs.push((name.to_string(), SessionInputValue::from(kv)));
                }
            } else {
                for j in 0..24 {
                    inputs.push((KV_NAMES[j][0].to_string(), SessionInputValue::from(&dummy_kvs[j * 2])));
                    inputs.push((KV_NAMES[j][1].to_string(), SessionInputValue::from(&dummy_kvs[j * 2 + 1])));
                }
            }

            let outputs = self.t3_session.run(inputs)?;
            
            let mut next_kvs = Vec::new();
            for j in 0..24 {
                let key_k = format!("present.{}.key", j);
                let val_k = outputs.get(&key_k).expect("Missing present.k");
                let (dim_k, data_k) = val_k.try_extract_tensor::<f32>()?;
                next_kvs.push(ort::value::Value::from_array((dim_k.to_owned(), data_k.to_vec()))?.into());
                
                let key_v = format!("present.{}.value", j);
                let val_v = outputs.get(&key_v).expect("Missing present.v");
                let (dim_v, data_v) = val_v.try_extract_tensor::<f32>()?;
                next_kvs.push(ort::value::Value::from_array((dim_v.to_owned(), data_v.to_vec()))?.into());
            }
            kv_cache = Some(next_kvs);

            let (dim, hidden_states) = outputs["last_hidden_state"].try_extract_tensor::<f32>()?;
            let seq_len = dim[1] as usize;
            let last_hidden = &hidden_states[(seq_len - 1) * 1024..];
            let last_hidden_arr = Array1::from_iter(last_hidden.iter().cloned());
            
            let mut logits = self.speech_head_weight.dot(&last_hidden_arr) + &self.speech_head_bias;
            for &prev_id in &speech_ids {
                let idx = prev_id as usize;
                if idx < logits.len() { logits[idx] -= 2.5; }
            }
            logits.mapv_inplace(|x| x / 0.3);

            let (next_token, _) = logits.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, val)| (idx as i64, val)).unwrap();

            if next_token == 6562 { break; }
            speech_ids.push(next_token);
        }
        Ok(speech_ids)
    }

    fn generate_mel(&mut self, tokens: &[i64]) -> Result<Array2<f32>> {
        let s3_tokens: Vec<i64> = tokens.iter().cloned().filter(|&t| t < 6561).collect();
        if s3_tokens.is_empty() { return Ok(Array2::zeros((80, 10))); }

        let tokens_arr = Array2::from_shape_vec((1, s3_tokens.len()), s3_tokens)?;
        let lens_arr = Array1::from_vec(vec![tokens_arr.shape()[1] as i64]);

        let enc_outputs = self.flow_encoder.run(vec![
            ("tokens", SessionInputValue::from(ort::value::Value::from_array(tokens_arr.clone())?)),
            ("lens", SessionInputValue::from(ort::value::Value::from_array(lens_arr.clone())?))
        ])?;
        
        let (mu_dim, mu) = enc_outputs["mu"].try_extract_tensor::<f32>()?;
        let mu_arr = Array3::from_shape_vec((mu_dim[0] as usize, mu_dim[1] as usize, mu_dim[2] as usize), mu.to_vec())?;
        
        let t_steps = 1;
        let dt = 1.0 / t_steps as f32;
        let mut x = Array2::zeros((80, mu_dim[2] as usize));
        let mut rng = rand::thread_rng();
        for val in x.iter_mut() { *val = rng.sample(StandardNormal); }

        for i in 0..t_steps {
            let t_arr = Array1::from_vec(vec![i as f32 * dt]);
            let r_arr = Array1::from_vec(vec![1.0f32]);
            
            let outputs = self.flow_estimator.run(vec![
                ("x", SessionInputValue::from(ort::value::Value::from_array(x.clone().insert_axis(Axis(0)))?)),
                ("mu", SessionInputValue::from(ort::value::Value::from_array(mu_arr.clone())?)),
                ("t", SessionInputValue::from(ort::value::Value::from_array(t_arr.clone())?)),
                ("speaker_emb", SessionInputValue::from(ort::value::Value::from_array(self.s3_speaker_emb.clone())?)),
                ("r", SessionInputValue::from(ort::value::Value::from_array(r_arr.clone())?)),
                ("cond", SessionInputValue::from(ort::value::Value::from_array(mu_arr.clone())?))
            ])?;
            
            let (v_dim, v) = outputs["velocity"].try_extract_tensor::<f32>()?;
            let v_arr = Array3::from_shape_vec((v_dim[0] as usize, v_dim[1] as usize, v_dim[2] as usize), v.to_vec())?;
            x += &(v_arr.slice(s![0, .., ..]).into_owned() * dt);
        }
        Ok(x)
    }

    fn generate_waveform(&mut self, mel: &Array2<f32>) -> Result<Vec<f32>> {
        let s_stft = Array3::<f32>::zeros((1, 18, 1));

        let outputs = self.vocoder_core.run(vec![
            ("mel", SessionInputValue::from(ort::value::Value::from_array(mel.clone().insert_axis(Axis(0)))?)),
            ("s_stft", SessionInputValue::from(ort::value::Value::from_array(s_stft)?))
        ])?;
        
        let (stft_dim, stft_out) = outputs["stft_out"].try_extract_tensor::<f32>()?;
        let stft_arr = Array3::from_shape_vec((stft_dim[0] as usize, stft_dim[1] as usize, stft_dim[2] as usize), stft_out.to_vec())?;
        let stft_view = stft_arr.slice(s![0, .., ..]);
        
        let n_fft = 16;
        let hop_len = 4;
        let n_bins = 9;
        let frames = stft_view.shape()[1];
        
        let mut overlap_buffer = vec![0.0f32; frames * hop_len + n_fft];
        let mut window = vec![0.0f32; n_fft];
        for i in 0..n_fft {
            window[i] = 0.5 * (1.0 - (2.0 * PI * i as f32 / n_fft as f32).cos());
        }

        let mut planner = FftPlanner::new();
        let ifft = planner.plan_fft_inverse(n_fft);

        for i in 0..frames {
            let mut spectrum = vec![Complex::new(0.0, 0.0); n_fft];
            for k in 0..n_bins {
                let mag = stft_view[[k, i]].exp().min(100.0);
                let p_val = stft_view[[k + n_bins, i]].sin();
                let c = Complex::new(mag * p_val.cos(), mag * p_val.sin());
                spectrum[k] = c;
                if k > 0 && k < n_bins - 1 { spectrum[n_fft - k] = c.conj(); }
            }
            ifft.process(&mut spectrum);
            let start = i * hop_len;
            for k in 0..n_fft {
                overlap_buffer[start + k] += (spectrum[k].re / n_fft as f32) * window[k];
            }
        }

        let peak = overlap_buffer.iter().map(|&x| x.abs()).fold(0.0, f32::max);
        if peak > 0.0 { 
            let scale = 0.8 / peak;
            overlap_buffer.iter_mut().for_each(|x| *x *= scale); 
        }
        Ok(overlap_buffer)
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let artifact_dir = Path::new("rust_agency/artifacts/chatterbox");
    if !artifact_dir.exists() { return Ok(()); }
    let mut turbo = ChatterboxTurbo::new(artifact_dir)?;
    let text = "I'm back and stable.";
    println!("Synthesizing: {}", text);
    let audio = turbo.generate(text)?;
    println!("Synthesis completed. Playing audio...");
    match rodio::OutputStream::try_default() {
        Ok((_stream, handle)) => {
            if let Ok(sink) = rodio::Sink::try_new(&handle) {
                let source = rodio::buffer::SamplesBuffer::new(1, 24000, audio);
                sink.append(source);
                sink.sleep_until_end();
            }
        }
        Err(e) => println!("Audio Playback Error: {}", e),
    }
    Ok(())
}