use std::borrow::Cow;

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
    data: &'d [u8],
}

pub struct RuntimeVertexBuilder<'d> {
    members: Vec<RuntimeVertexMember<'d>>,
    offset: usize,
}

impl<'d> RuntimeVertexBuilder<'d> {
    #[inline]
    pub fn new() -> Self {
        Self {
            members: Vec::new(),
            offset: 0,
        }
    }

    #[inline]
    pub fn add<T>(mut self, attribute: VertexAttribute, data: &'d [T]) -> Self
    where
        [T]: BufferContents,
        T: Pod,
    {
        let field_size = std::mem::size_of::<T>() as u32;
        let format_size = attribute
            .format
            .block_size()
            .expect("no block size for format") as u32;
        let num_elements = field_size / format_size;
        let remainder = field_size % format_size;
        assert!(
            remainder == 0,
            "type of buffer elements for attribute `{}` does not fit provided format `{:?}`",
            attribute.name,
            attribute.format,
        );
        self.members.push(RuntimeVertexMember {
            names: vec![attribute.name], // TODO: support multiple names?
            info: VertexMemberInfo {
                offset: self.offset,
                format: attribute.format,
                num_elements,
            },
            data: data.as_bytes(),
        });
        self.offset += field_size as usize;

        self
    }

    #[inline]
    pub fn build(self) -> Option<(RuntimeVertexIter<'d>, VertexBufferInfo)> {
        // TODO: return Result instead!
        // Let check if all buffers have the same number of elements
        let mut num_vertices = 0;
        for member in &self.members {
            let field_size = member.info.num_elements
                * member
                    .info
                    .format
                    .block_size()
                    .expect("no block size for format") as u32;
            if num_vertices == 0 {
                num_vertices = member.data.len() / field_size as usize;
            } else if num_vertices != (member.data.len() / field_size as usize) {
                return None;
            }
        }

        let info = VertexBufferInfo {
            members: self
                .members
                .iter()
                .map(|member| (member.names[0].to_string(), member.info.clone()))
                .collect(),
            stride: self.offset as u32,
            input_rate: super::VertexInputRate::Vertex,
        };

        let length = self.members.iter().map(|member| member.data.len()).sum();
        // We need to know the byte ranges of the vertex that belong to our members
        let mut member_max: Vec<usize> = self
            .members
            .iter()
            .skip(1)
            .map(|member| member.info.offset)
            .collect();
        member_max.push(self.offset); // stride
        let member_min: Vec<usize> = self
            .members
            .iter()
            .map(|member| member.info.offset)
            .collect();
        let member_ranges = member_min.into_iter().zip(member_max.into_iter()).collect();

        let iter = RuntimeVertexIter {
            members: self.members,
            member_ranges,
            stride: self.offset as u32,
            length,
            index: 0,
        };

        Some((iter, info))
    }
}

pub struct RuntimeVertexIter<'d> {
    members: Vec<RuntimeVertexMember<'d>>,
    member_ranges: Vec<(usize, usize)>,
    stride: u32,
    length: usize,
    index: usize,
}

impl<'d> Iterator for RuntimeVertexIter<'d> {
    type Item = u8;

    fn next(&mut self) -> Option<Self::Item> {
        if self.length == self.index {
            return None;
        }
        let vertex_index = self.index / (self.stride as usize);
        let data_offset = self.index % (self.stride as usize);
        let member_index = self
            .member_ranges
            .iter()
            .position(|range| range.0 <= data_offset && range.1 > data_offset)
            .unwrap();
        let member = &self.members[member_index];
        let field_size = self.member_ranges[member_index].1 - self.member_ranges[member_index].0;
        let member_offset = data_offset - self.member_ranges[member_index].0;
        let data = member.data[vertex_index * field_size + member_offset];
        self.index += 1;
        Some(data)
    }
}

impl<'d> ExactSizeIterator for RuntimeVertexIter<'d> {
    fn len(&self) -> usize {
        self.length - self.index
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
