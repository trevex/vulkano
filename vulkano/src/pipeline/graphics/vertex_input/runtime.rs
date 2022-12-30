use std::borrow::Cow;
use std::iter::{Cloned, Enumerate, FlatMap};
use std::vec::IntoIter;

use bytemuck::Pod;

use crate::{buffer::BufferContents, format::Format};

use super::{VertexBufferInfo, VertexMemberInfo};

#[derive(Hash, Eq, PartialEq, Debug)]
pub struct VertexAttribute {
    pub name: Cow<'static, str>, // TODO: support multiple names?
    pub format: Format,
}

impl VertexAttribute {
    #[inline]
    pub const fn new(name: &'static str, format: Format) -> Self {
        Self {
            name: Cow::Borrowed(name),
            format,
        }
    }
}

struct RuntimeVertexMember<'d> {
    names: Vec<Cow<'static, str>>,
    info: VertexMemberInfo,
    field_size: usize,
    data: &'d [u8],
}

pub struct RuntimeVertexBuilder<'d> {
    members: Vec<(Cow<'static, str>, VertexMemberInfo)>,
    slices: Vec<(&'d [u8], usize)>,
    offset: usize,
}

impl<'d> RuntimeVertexBuilder<'d> {
    #[inline]
    pub fn new() -> Self {
        Self {
            members: Vec::new(),
            slices: Vec::new(),
            offset: 0,
        }
    }

    #[inline]
    pub fn add<T>(mut self, attribute: VertexAttribute, data: &'d [T]) -> Self
    where
        [T]: BufferContents,
        T: Pod,
    {
        let field_size = std::mem::size_of::<T>();
        let format_size = attribute
            .format
            .block_size()
            .expect("no block size for format") as usize;
        let num_elements = field_size / format_size;
        let remainder = field_size % format_size;
        assert!(
            remainder == 0,
            "type of buffer elements for attribute `{}` does not fit provided format `{:?}`",
            attribute.name,
            attribute.format,
        );

        self.members.push((
            attribute.name, // TODO: support multiple names?
            VertexMemberInfo {
                offset: self.offset,
                format: attribute.format,
                num_elements: num_elements as u32,
            },
        ));
        self.offset += field_size;

        self.slices.push((data.as_bytes(), field_size));

        self
    }

    #[inline]
    pub fn build(
        self,
    ) -> Option<(
        // Cloned<
        //     FlatMap<
        //         Enumerate<IntoIter<(&'d [u8], usize)>>,
        //         &'d [u8],
        //         impl FnMut((usize, (&'d [u8], usize))) -> &'d [u8],
        //     >,
        // >,
        // Cloned<
        //     FlatMap<
        //         std::ops::Range<usize>,
        //         FlatMap<
        //             core::slice::Iter<'d, (&'d [u8], usize)>,
        //             &'d [u8],
        //             impl FnMut(&'d (&'d [u8], usize)) -> &'d [u8],
        //         >,
        //         impl FnMut(
        //             usize,
        //         ) -> FlatMap<
        //             core::slice::Iter<'d, (&'d [u8], usize)>,
        //             &'d [u8],
        //             impl FnMut(&'d (&'d [u8], usize)) -> &'d [u8],
        //         >,
        //     >,
        // >,
        impl Iterator<Item = u8> + 'd,
        VertexBufferInfo,
    )> {
        // TODO: return Result instead!

        let info = VertexBufferInfo {
            members: self
                .members
                .iter()
                .map(|member| (member.0.to_string(), member.1.clone()))
                .collect(),
            stride: self.offset as u32,
            input_rate: super::VertexInputRate::Vertex,
        };

        let count = self
            .slices
            .iter()
            .map(|(data, size)| data.len() / size)
            .min()
            .unwrap();
        let iter = (0..count)
            .zip(std::iter::repeat(self.slices))
            .flat_map(move |(i, slices)| {
                slices
                    .iter()
                    .flat_map(|(data, size)| &data[i * size..i * (size + 1)])
                    .collect::Vec<u8>()
            })
            .cloned();
        // let iter = attributes
        //     .into_iter()
        //     .enumerate()
        //     .flat_map(move |(i, (member, size))| &member.data[i * size..i * (size + 1)])
        //     .cloned();

        Some((iter, info))
    }
}

#[cfg(test)]
mod tests {
    use bytemuck::{Pod, Zeroable};

    use crate::{
        buffer::BufferContents, format::Format,
        pipeline::graphics::vertex_input::runtime::RuntimeVertexBuilder,
    };

    use super::VertexAttribute;

    const ATTRIBUTE_POSITION: VertexAttribute =
        VertexAttribute::new("position", Format::R32G32B32_SFLOAT);
    const ATTRIBUTE_UVS: VertexAttribute = VertexAttribute::new("uvs", Format::R32G32_SFLOAT);

    #[test]
    fn runtime_vertex_builder() {
        let pos_0 = [0.1f32, 1.2, 2.3];
        let pos_1 = [3.4f32, 4.5, 5.6];
        let positions = [pos_0, pos_1];
        #[repr(C)]
        #[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
        struct Vec2 {
            x: f32,
            y: f32,
        }
        let uv_0 = Vec2 { x: 0.15, y: 1.0 };
        let uv_1 = Vec2 { x: 0.72, y: 0.0 };
        let uvs = [uv_0, uv_1];

        let (iter, info) = RuntimeVertexBuilder::new()
            .add(ATTRIBUTE_POSITION, &positions)
            .add(ATTRIBUTE_UVS, &uvs)
            .build()
            .unwrap();

        let data: Vec<u8> = iter.collect();
        let mut expected = pos_0.as_bytes().to_vec();
        expected.append(&mut uv_0.as_bytes().to_vec());
        expected.append(&mut pos_1.as_bytes().to_vec());
        expected.append(&mut uv_1.as_bytes().to_vec());

        assert_eq!(info.stride, 3 * 4 + 2 * 4);
        assert_eq!(data.len(), expected.len());
        assert_eq!(
            data.len(),
            data.iter()
                .zip(expected.iter())
                .filter(|&(a, b)| a == b)
                .count()
        );
    }
}
