use super::{ExecutionProvider, RegisterError};
use crate::{error::Result, ortsys, session::builder::SessionBuilder};

/// [MPS execution provider](https://onnxruntime.ai/docs/execution-providers/MPS-ExecutionProvider.html)
/// for Apple Silicon GPUs.
#[derive(Debug, Default, Clone)]
pub struct MPS {
	flags: u32
}

super::impl_ep!(MPS);

impl MPS {
	/// Configures the MPS execution provider flags.
	#[must_use]
	pub fn with_flags(mut self, flags: u32) -> Self {
		self.flags = flags;
		self
	}
}

impl ExecutionProvider for MPS {
	fn name(&self) -> &'static str {
		"MPSExecutionProvider"
	}

	fn supported_by_platform(&self) -> bool {
		cfg!(all(target_vendor = "apple", target_arch = "aarch64"))
	}

	#[allow(unused, unreachable_code)]
	fn register(&self, session_builder: &mut SessionBuilder) -> Result<(), RegisterError> {
		#[cfg(any(feature = "load-dynamic", feature = "mps"))]
		{
			use crate::AsPointer;

			ortsys![unsafe SessionOptionsAppendExecutionProvider_MPS(session_builder.ptr_mut(), self.flags)?];
			return Ok(());
		}

		Err(RegisterError::MissingFeature)
	}
}
