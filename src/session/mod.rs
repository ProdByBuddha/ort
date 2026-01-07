use alloc::{
	borrow::Cow,
	sync::Arc,
	vec,
	vec::Vec
};
use core::{
	any::Any,
	cell::UnsafeCell,
	fmt::{self, Debug},
	iter,
	marker::PhantomData,
	ptr::{self, NonNull},
	slice
};

use smallvec::SmallVec;

#[cfg(feature = "ndarray")]
use ndarray::ArrayD;

use crate::{
	AsPointer,
	char_p_to_str,
	error::{Error, ErrorCode, Result},
	io_binding::IoBinding,
	memory::Allocator,
	ortsys,
	session::{
		SharedSessionInner,
		output::SessionOutputs,
		run_options::{NoSelectedOutputs, RunOptions, SelectedOutputMarker, UntypedRunOptions}
	},
	util::stack::{STACK_SESSION_INPUTS, STACK_SESSION_OUTPUTS},
	value::{DynValue, DynValueTypeMarker, Value, ValueInner, ValueType}
};

pub mod builder;
pub mod input;
pub mod output;
pub mod run_options;

use self::input::{SessionInputs, SessionInputValue};

/// An ONNX Runtime session.
pub struct Session {
	pub(crate) inner: Arc<SharedSessionInner>,
	pub inputs: Vec<input::Input>,
	pub outputs: Vec<output::Output>
}

unsafe impl Send for Session {}
unsafe impl Sync for Session {}

impl Session {
	pub fn builder() -> Result<builder::SessionBuilder> {
		builder::SessionBuilder::new()
	}

	pub fn inputs(&self) -> &[input::Input] {
		&self.inputs
	}

	pub fn outputs(&self) -> &[output::Output] {
		&self.outputs
	}

	pub fn run<'s, 'i, 'v: 'i, const N: usize>(&'s mut self, input_values: impl Into<SessionInputs<'i, 'v, N>>) -> Result<SessionOutputs<'s>> {
		match input_values.into() {
			SessionInputs::ValueSlice(input_values) => {
				self.run_inner(self.inputs.iter().map(|input| input.name()).collect(), input_values.iter().collect(), None)
			}
			SessionInputs::ValueArray(input_values) => {
				self.run_inner(self.inputs.iter().map(|input| input.name()).collect(), input_values.iter().collect(), None)
			}
			SessionInputs::ValueMap(input_values) => {
				self.run_inner(input_values.iter().map(|(k, _)| k.as_ref()).collect(), input_values.iter().map(|(_, v)| v).collect(), None)
			}
		}
	}

	pub fn run_with_options<'r, 's: 'r, 'i, 'v: 'i, O: SelectedOutputMarker, const N: usize>(
		&'s mut self,
		input_values: impl Into<SessionInputs<'i, 'v, N>>,
		run_options: &'r RunOptions<O>
	) -> Result<SessionOutputs<'r>> {
		match input_values.into() {
			SessionInputs::ValueSlice(input_values) => {
				self.run_inner(self.inputs.iter().map(|input| input.name()).collect(), input_values.iter().collect(), Some(&run_options.inner))
			}
			SessionInputs::ValueArray(input_values) => {
				self.run_inner(self.inputs.iter().map(|input| input.name()).collect(), input_values.iter().collect(), Some(&run_options.inner))
			}
			SessionInputs::ValueMap(input_values) => {
				self.run_inner(input_values.iter().map(|(k, _)| k.as_ref()).collect(), input_values.iter().map(|(_, v)| v).collect(), Some(&run_options.inner))
			}
		}
	}

	fn run_inner<'i, 'r, 's: 'r, 'v: 'i>(
		&'s self,
		input_names: SmallVec<[&str; STACK_SESSION_INPUTS]>,
		input_values: SmallVec<[&'i SessionInputValue<'v>; STACK_SESSION_INPUTS]>,
		run_options: Option<&'r UntypedRunOptions>
	) -> Result<SessionOutputs<'r>> {
		if input_values.len() > input_names.len() {
			return Err(Error::new_with_code(
				ErrorCode::InvalidArgument,
				format!("{} inputs were provided, but the model only accepts {}.", input_values.len(), input_names.len())
			));
		}

		let (output_names, mut output_tensors) = match run_options {
			Some(r) => r.outputs.resolve_outputs(&self.outputs),
			None => (self.outputs.iter().map(|o| o.name()).collect(), iter::repeat_with(|| None).take(self.outputs.len()).collect())
		};
		let mut output_value_ptrs: Vec<*mut ort_sys::OrtValue> = output_tensors
			.iter_mut()
			.map(|c| match c {
				Some(v) => v.ptr_mut(),
				None => ptr::null_mut()
			})
			.collect();
		let input_value_ptrs: Vec<*const ort_sys::OrtValue> = input_values.iter().map(|c| c.ptr()).collect();

		let run_options_ptr = if let Some(run_options) = &run_options { run_options.ptr.as_ptr() } else { ptr::null() };

		let input_name_cstrs: Vec<std::ffi::CString> = input_names.iter().map(|s| std::ffi::CString::new(s.as_bytes()).unwrap()).collect();
		let input_name_ptrs: Vec<*const core::ffi::c_char> = input_name_cstrs.iter().map(|s| s.as_ptr()).collect();
		let output_name_cstrs: Vec<std::ffi::CString> = output_names.iter().map(|s| std::ffi::CString::new(s.as_bytes()).unwrap()).collect();
		let output_name_ptrs: Vec<*const core::ffi::c_char> = output_name_cstrs.iter().map(|s| s.as_ptr()).collect();

		ortsys![
			unsafe Run(
				self.inner.session_ptr.as_ptr(),
				run_options_ptr,
				input_name_ptrs.as_ptr(),
				input_value_ptrs.as_ptr(),
				input_value_ptrs.len(),
				output_name_ptrs.as_ptr(),
				output_name_ptrs.len(),
				output_value_ptrs.as_mut_ptr()
			)?
		];

		let outputs = output_tensors
			.into_iter()
			.enumerate()
			.map(|(i, v)| match v {
				Some(value) => value,
				None => unsafe {
					Value::from_ptr(
						NonNull::new(output_value_ptrs[i]).expect("OrtValue ptr returned from session Run should not be null"),
						Some(Arc::clone(&self.inner))
					)
				}
			})
			.collect();

		Ok(SessionOutputs::new(output_names, outputs))
	}

	#[cfg(not(target_arch = "wasm32"))]
	pub fn run_binding<'b, 's: 'b>(&'s mut self, binding: &'b IoBinding) -> Result<SessionOutputs<'b>> {
		self.run_binding_inner(binding, None)
	}

	#[cfg(not(target_arch = "wasm32"))]
	fn run_binding_inner<'r, 'b, 's: 'b>(
		&'s self,
		binding: &'b IoBinding,
		run_options: Option<&'r RunOptions<NoSelectedOutputs>>
	) -> Result<SessionOutputs<'b>> {
		let run_options_ptr = if let Some(run_options) = run_options { run_options.ptr() } else { ptr::null() };
		ortsys![unsafe RunWithBinding(self.inner.ptr().cast_mut(), run_options_ptr, binding.ptr())?];

		let mut count = binding.output_values.len();
		if count > 0 {
			let mut output_values_ptr: *mut *mut ort_sys::OrtValue = ptr::null_mut();
			ortsys![unsafe GetBoundOutputValues(binding.ptr(), self.inner.allocator.ptr().cast_mut(), &mut output_values_ptr, &mut count)?; nonNull(output_values_ptr)];

			let output_values = unsafe { slice::from_raw_parts(output_values_ptr.as_ptr(), count) }
				.iter()
				.map(|ptr| unsafe {
					DynValue::from_ptr(NonNull::new(*ptr).expect("OrtValue ptrs returned by GetBoundOutputValues should not be null"), Some(self.inner.clone()))
				})
				.collect();
			unsafe {
				self.inner.allocator.free(output_values_ptr.as_ptr());
			}

			Ok(SessionOutputs::new(binding.output_values.iter().map(|(k, _)| k.as_str()).collect(), output_values))
		} else {
			Ok(SessionOutputs::new_empty())
		}
	}

	#[inline]
	pub fn inner(&self) -> Arc<SharedSessionInner> {
		Arc::clone(&self.inner)
	}
}

impl AsPointer for Session {
	type Ptr = ort_sys::OrtSession;
	fn ptr(&self) -> *const Self::Ptr {
		self.inner.session_ptr.as_ptr()
	}
}
