use std::borrow::Cow;

use bytemuck::Pod;

use crate::{buffer::BufferContents, format::Format};

use super::{VertexBufferInfo, VertexMemberInfo};

#[derive(Hash, Eq, PartialEq, Debug)]
pub struct VertexAttribute {
    pub name: Cow<'static, str>,
    pub format: Format, // TODO: specify num_elements as well!
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
            attribute.name,
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
    pub fn build(self) -> (RuntimeVertexIter<'d>, VertexBufferInfo) {
        // TODO: return Result instead!
        let num_vertices = self
            .slices
            .iter()
            .map(|(data, size)| data.len() / size)
            .min()
            .unwrap();

        let info = VertexBufferInfo {
            members: self
                .members
                .iter()
                .map(|member| (member.0.to_string(), member.1.clone()))
                .collect(),
            stride: self.offset as u32,
            input_rate: super::VertexInputRate::Vertex,
        };

        let data_length = self
            .slices
            .iter()
            .map(|(_data, size)| size * num_vertices)
            .sum();

        // We need to know the byte ranges of the vertex that belong to our members
        let member_max = self
            .members
            .iter()
            .skip(1)
            .map(|member| member.1.offset)
            .chain([self.offset].into_iter());
        let member_min = self.members.iter().map(|member| member.1.offset);
        let member_ranges = member_min.zip(member_max).collect();

        let iter = RuntimeVertexIter {
            member_ranges,
            member_slices: self.slices,
            stride: self.offset as u32,
            data_length,
            data_index: 0,
            member_index: 0,
        };

        (iter, info)
    }
}

pub struct RuntimeVertexIter<'d> {
    member_ranges: Vec<(usize, usize)>,
    member_slices: Vec<(&'d [u8], usize)>,
    stride: u32,
    data_length: usize,
    data_index: usize,
    member_index: usize,
}

impl<'d> Iterator for RuntimeVertexIter<'d> {
    type Item = u8;

    fn next(&mut self) -> Option<Self::Item> {
        if self.data_length == self.data_index {
            return None;
        }
        let vertex_index = self.data_index / (self.stride as usize);
        let data_offset = self.data_index % (self.stride as usize);
        if self.member_ranges[self.member_index].1 <= data_offset
            || self.member_ranges[self.member_index].0 > data_offset
        {
            self.member_index += 1;
            if self.member_index == self.member_ranges.len() {
                self.member_index = 0;
            }
        }
        let field_size =
            self.member_ranges[self.member_index].1 - self.member_ranges[self.member_index].0;
        let member_offset = data_offset - self.member_ranges[self.member_index].0;
        let data = self.slices[self.member_index].0[vertex_index * field_size + member_offset];
        self.data_index += 1;
        Some(data)
    }
}

impl<'d> ExactSizeIterator for RuntimeVertexIter<'d> {
    fn len(&self) -> usize {
        self.data_length - self.data_index
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
            .build();

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

    use test::Bencher;

    #[bench]
    fn bench_runtime_vertex(b: &mut Bencher) {
        let pos_0 = [0.1f32, 1.2, 2.3];
        let pos_1 = [3.4f32, 4.5, 5.6];
        let positions = [
            pos_0, pos_1, pos_0, pos_1, pos_0, pos_1, pos_0, pos_1, pos_0, pos_1, pos_0, pos_1,
            pos_0, pos_1, pos_0, pos_1, pos_0, pos_1, pos_0, pos_1, pos_0, pos_1, pos_0, pos_1,
        ];
        #[repr(C)]
        #[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
        struct Vec2 {
            x: f32,
            y: f32,
        }
        let uv_0 = Vec2 { x: 0.15, y: 1.0 };
        let uv_1 = Vec2 { x: 0.72, y: 0.0 };
        let uvs = [
            uv_0, uv_1, uv_0, uv_1, uv_0, uv_1, uv_0, uv_1, uv_0, uv_1, uv_0, uv_1, uv_0, uv_1,
            uv_0, uv_1, uv_0, uv_1, uv_0, uv_1, uv_0, uv_1, uv_0, uv_1,
        ];
        b.iter(|| {
            let (iter, _info) = RuntimeVertexBuilder::new()
                .add(ATTRIBUTE_POSITION, &positions)
                .add(ATTRIBUTE_UVS, &uvs)
                .build();

            iter.collect::<Vec<u8>>()
        })
    }
}
