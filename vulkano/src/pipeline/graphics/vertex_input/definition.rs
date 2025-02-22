// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Definition for creating a [`VertexInputState`] based on a [`ShaderInterface`].
//!
//! # Implementing `Vertex`
//!
//! The implementations of the `VertexDefinition` trait that are provided by vulkano require you to
//! use a buffer whose content is `[V]` where `V` implements the `Vertex` trait.
//!
//! The `Vertex` trait is unsafe, but can be implemented on a struct with the `impl_vertex!`
//! macro.
//!
//! # Examples
//!
//! ```ignore       // TODO:
//! # #[macro_use] extern crate vulkano
//! # fn main() {
//! # use std::sync::Arc;
//! # use vulkano::device::Device;
//! # use vulkano::device::Queue;
//! use vulkano::buffer::BufferAccess;
//! use vulkano::buffer::BufferUsage;
//! use vulkano::memory::HostVisible;
//! use vulkano::pipeline::vertex::;
//! # let device: Arc<Device> = return;
//! # let queue: Arc<Queue> = return;
//!
//! struct Vertex {
//!     position: [f32; 2]
//! }
//!
//! impl_vertex!(Vertex, position);
//!
//! let usage = BufferUsage {
//!     vertex_buffer: true,
//!     ..BufferUsage::empty()
//! };
//!
//! let vertex_buffer = BufferAccess::<[Vertex], _>::array(&device, 128, &usage, HostVisible, &queue)
//!     .expect("failed to create buffer");
//!
//! // TODO: finish example
//! # }
//! ```

use crate::{
    pipeline::graphics::vertex_input::{VertexInputState, VertexMemberInfo},
    shader::{ShaderInterface, ShaderInterfaceEntryType},
};
use std::{
    error::Error,
    fmt::{Display, Error as FmtError, Formatter},
};

/// Trait for types that can create a [`VertexInputState`] from a [`ShaderInterface`].
pub unsafe trait VertexDefinition {
    /// Builds the vertex definition to use to link this definition to a vertex shader's input
    /// interface.
    // TODO: remove error return, move checks to GraphicsPipelineBuilder
    fn definition(
        &self,
        interface: &ShaderInterface,
    ) -> Result<VertexInputState, IncompatibleVertexDefinitionError>;
}

unsafe impl VertexDefinition for VertexInputState {
    fn definition(
        &self,
        _interface: &ShaderInterface,
    ) -> Result<VertexInputState, IncompatibleVertexDefinitionError> {
        Ok(self.clone())
    }
}

/// Error that can happen when the vertex definition doesn't match the input of the vertex shader.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IncompatibleVertexDefinitionError {
    /// An attribute of the vertex shader is missing in the vertex source.
    MissingAttribute {
        /// Name of the missing attribute.
        attribute: String,
    },

    /// The format of an attribute does not match.
    FormatMismatch {
        /// Name of the attribute.
        attribute: String,
        /// The format in the vertex shader.
        shader: ShaderInterfaceEntryType,
        /// The format in the vertex definition.
        definition: VertexMemberInfo,
    },
}

impl Error for IncompatibleVertexDefinitionError {}

impl Display for IncompatibleVertexDefinitionError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        match self {
            IncompatibleVertexDefinitionError::MissingAttribute { .. } => {
                write!(f, "an attribute is missing")
            }
            IncompatibleVertexDefinitionError::FormatMismatch { .. } => {
                write!(f, "the format of an attribute does not match")
            }
        }
    }
}
