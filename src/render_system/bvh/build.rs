use nalgebra::{matrix, vector, Isometry3, Point3, Vector3, Vector6};

use crate::{
    render_system::bvh::{aabb::Aabb, BvhNode},
    utils,
};

use super::super::vertex::Vertex3D;

#[derive(Clone, Debug)]
struct BuildBvhLeaf {
    parent_idx: usize,
    first_prim_idx_idx: usize,
    prim_count: usize,
}

#[derive(Clone, Debug)]
struct BuildBvhInternalNode {
    parent_idx: usize,
    left_child_idx: usize,
    right_child_idx: usize,
}

#[derive(Clone, Debug)]
enum BuildBvhNodeKind {
    Leaf(BuildBvhLeaf),
    InternalNode(BuildBvhInternalNode),
}

#[derive(Clone, Debug)]
struct BuildBvhNode {
    aabb: Aabb,
    kind: BuildBvhNodeKind,
}

fn blas_leaf_bounds(leaf: &BuildBvhLeaf, prim_idxs: &[usize], prim_aabbs: &[Aabb]) -> Aabb {
    let mut bound = Aabb::Empty;
    for i in leaf.first_prim_idx_idx..(leaf.first_prim_idx_idx + leaf.prim_count) {
        let prim_aabb = prim_aabbs[prim_idxs[i]];
        bound = Aabb::union(&bound, &prim_aabb);
    }
    bound
}

fn find_best_plane(
    leaf: &BuildBvhLeaf,
    prim_idxs: &[usize],
    prim_centroids: &[Point3<f32>],
    prim_aabbs: &[Aabb],
    cost_function: &impl Fn(&Aabb, &Aabb, usize, usize) -> f32,
) -> (usize, f32) {
    const BINS: usize = 32;

    let mut best_cost = f32::MAX;
    let mut best_dimension = 0;
    let mut best_split_pos = 0.0;

    for dimension in 0..3 {
        // find the bounds over the centroids of all the primitives
        let mut bounds_min = f32::MAX;
        let mut bounds_max = f32::MIN;
        for i in leaf.first_prim_idx_idx..(leaf.first_prim_idx_idx + leaf.prim_count) {
            let centroid = &prim_centroids[prim_idxs[i]];
            bounds_min = bounds_min.min(centroid[dimension]);
            bounds_max = bounds_max.max(centroid[dimension]);
        }

        // the bounding box of each bin
        let mut bin_bounds = [Aabb::Empty; BINS];
        // the number of primitives in each bin
        let mut bin_primcount = [0; BINS];

        // assign each triangle to a bin
        let scale = BINS as f32 / (bounds_max - bounds_min);
        for i in leaf.first_prim_idx_idx..(leaf.first_prim_idx_idx + leaf.prim_count) {
            let prim_idx = prim_idxs[i];
            let prim_aabb = &prim_aabbs[prim_idx];
            let prim_centroid = &prim_centroids[prim_idx];
            let bin_idx = usize::min(
                BINS - 1,
                ((prim_centroid[dimension] - bounds_min) * scale) as usize,
            );
            bin_primcount[bin_idx] += 1;
            bin_bounds[bin_idx] = Aabb::union(&bin_bounds[bin_idx], &prim_aabb);
        }

        // there are BINS - 1 possible splits
        // 1 plane between every two bins
        let mut plane_aabb_to_left = [Aabb::Empty; BINS - 1];
        let mut plane_aabb_to_right = [Aabb::Empty; BINS - 1];
        let mut plane_primcount_to_left = [0; BINS - 1];
        let mut plane_primcount_to_right = [0; BINS - 1];

        let mut aabb_to_left = Aabb::Empty;
        let mut aabb_to_right = Aabb::Empty;
        let mut primcount_to_left = 0;
        let mut primcount_to_right = 0;

        for plane in 0..(BINS - 1) {
            primcount_to_left += bin_primcount[plane];
            plane_primcount_to_left[plane] = primcount_to_left;
            aabb_to_left = Aabb::union(&aabb_to_left, &bin_bounds[plane]);
            plane_aabb_to_left[plane] = aabb_to_left;

            primcount_to_right += bin_primcount[BINS - 1 - plane];
            plane_primcount_to_right[BINS - 2 - plane] = primcount_to_right;
            aabb_to_right = Aabb::union(&aabb_to_right, &bin_bounds[BINS - 1 - plane]);
            plane_aabb_to_right[BINS - 2 - plane] = aabb_to_right;
        }

        let scale = (bounds_max - bounds_min) / BINS as f32;

        for plane in 0..(BINS - 1) {
            let cost = cost_function(
                &plane_aabb_to_left[plane],
                &plane_aabb_to_right[plane],
                plane_primcount_to_left[plane],
                plane_primcount_to_right[plane],
            );
            if cost < best_cost {
                best_cost = cost;
                best_dimension = dimension;
                best_split_pos = bounds_min + (plane as f32 + 1.0) * scale;
            }
        }
    }
    (best_dimension, best_split_pos)
}

fn subdivide(
    node_idx: usize,
    prim_idxs: &mut [usize],
    prim_aabbs: &[Aabb],
    prim_centroids: &[Point3<f32>],
    nodes: &mut Vec<BuildBvhNode>,
    cost_function: &impl Fn(&Aabb, &Aabb, usize, usize) -> f32,
) {
    match nodes[node_idx].kind {
        BuildBvhNodeKind::Leaf(ref leaf) if leaf.prim_count > 2 => {
            // get best plane to split along
            let (dimension, split_pos) =
                find_best_plane(leaf, prim_idxs, prim_centroids, prim_aabbs, cost_function);

            // partition the primitives in place by modifying prim_idxs
            let mut partitions = partition::partition(
                &mut prim_idxs
                    [leaf.first_prim_idx_idx..(leaf.first_prim_idx_idx + leaf.prim_count)],
                |&prim_idx| prim_centroids[prim_idx][dimension] < split_pos,
            );

            // If one of the subdivisions is empty then we fall back to randomly partitioning
            if partitions.0.len() == 0 || partitions.1.len() == 0 {
                dbg!("Falling back to random partitioning");
                partitions = prim_idxs
                    [leaf.first_prim_idx_idx..(leaf.first_prim_idx_idx + leaf.prim_count)]
                    .split_at_mut(leaf.prim_count / 2)
            }

            // create left child
            let left_leaf = BuildBvhLeaf {
                parent_idx: node_idx,
                first_prim_idx_idx: leaf.first_prim_idx_idx,
                prim_count: partitions.0.len(),
            };

            // create right child
            let right_leaf = BuildBvhLeaf {
                parent_idx: node_idx,
                first_prim_idx_idx: leaf.first_prim_idx_idx + partitions.0.len(),
                prim_count: partitions.1.len(),
            };

            let parent_idx = leaf.parent_idx;

            // insert children
            let left_child_idx = insert_blas_leaf_node(left_leaf, nodes, prim_idxs, prim_aabbs);
            let right_child_idx = insert_blas_leaf_node(right_leaf, nodes, prim_idxs, prim_aabbs);

            // recurse
            subdivide(
                left_child_idx,
                prim_idxs,
                prim_aabbs,
                prim_centroids,
                nodes,
                cost_function,
            );
            subdivide(
                right_child_idx,
                prim_idxs,
                prim_aabbs,
                prim_centroids,
                nodes,
                cost_function,
            );

            nodes[node_idx].kind = BuildBvhNodeKind::InternalNode(BuildBvhInternalNode {
                parent_idx,
                left_child_idx,
                right_child_idx,
            });
        }
        BuildBvhNodeKind::Leaf(ref leaf) if leaf.prim_count == 2 => {
            // create left child
            let left_leaf = BuildBvhLeaf {
                parent_idx: node_idx,
                first_prim_idx_idx: leaf.first_prim_idx_idx,
                prim_count: 1,
            };

            // create right child
            let right_leaf = BuildBvhLeaf {
                parent_idx: node_idx,
                first_prim_idx_idx: leaf.first_prim_idx_idx + 1,
                prim_count: 1,
            };

            let parent_idx = leaf.parent_idx;

            // insert children
            let left_child_idx = insert_blas_leaf_node(left_leaf, nodes, prim_idxs, prim_aabbs);
            let right_child_idx = insert_blas_leaf_node(right_leaf, nodes, prim_idxs, prim_aabbs);

            // update parent
            nodes[node_idx].kind = BuildBvhNodeKind::InternalNode(BuildBvhInternalNode {
                parent_idx,
                left_child_idx,
                right_child_idx,
            });
        }
        _ => {}
    }
}

fn insert_blas_leaf_node(
    leaf: BuildBvhLeaf,
    nodes: &mut Vec<BuildBvhNode>,
    prim_idxs: &[usize],
    prim_aabbs: &[Aabb],
) -> usize {
    let node_idx = nodes.len();
    nodes.push(BuildBvhNode {
        aabb: blas_leaf_bounds(&leaf, prim_idxs, prim_aabbs),
        kind: BuildBvhNodeKind::Leaf(leaf),
    });
    node_idx
}

pub fn build_bl_bvh(
    // the power/area of each primitive
    prim_luminances: &[f32],
    // prim triangles
    prim_vertexes: &[Point3<f32>],
    // if this is a bottom level bvh, then this is the primitive index of each primitive
    prim_index_ids: &[u32],
) -> (Vec<BvhNode>, Aabb, f32) {
    let n_prims = prim_luminances.len();
    assert_eq!(n_prims, prim_index_ids.len());

    let mut prim_idxs = (0..n_prims).collect::<Vec<_>>();

    let prim_aabbs = prim_vertexes
        .array_chunks()
        .map(|chunk: &[Point3<f32>; 3]| Aabb::from_points(chunk))
        .collect::<Vec<_>>();

    let prim_centroids = prim_vertexes
        .array_chunks()
        .map(|[v0, v1, v2]| Point3::from((v0.coords + v1.coords + v2.coords) / 3.0))
        .collect::<Vec<_>>();

    let mut nodes = vec![];

    // create root node
    let root_node_idx = insert_blas_leaf_node(
        BuildBvhLeaf {
            parent_idx: u32::MAX as usize,
            first_prim_idx_idx: 0,
            prim_count: n_prims,
        },
        &mut nodes,
        &prim_idxs,
        &prim_aabbs,
    );

    // surface area metric
    fn cost_function(aabb1: &Aabb, aabb2: &Aabb, count1: usize, count2: usize) -> f32 {
        aabb1.area() * count1 as f32 + aabb2.area() * count2 as f32
    }

    subdivide(
        root_node_idx,
        &mut prim_idxs,
        &prim_aabbs,
        &prim_centroids,
        &mut nodes,
        &cost_function,
    );

    let padding = vector![0.0001, 0.0001, 0.0001];

    // mapping from opt_bvh index to prim index
    let mut opt_bvh_idx_to_prim_idx = vec![];

    // nodes now contains a list of all the nodes in the blas.
    // however, it contains rust constructs and is not able to be passed to the shader
    // we now need to convert it into the finalized state that is optimized for gpu consumption
    let mut opt_bvh = nodes
        .into_iter()
        .map(|node| match node.kind {
            BuildBvhNodeKind::Leaf(ref leaf) => {
                assert!(leaf.prim_count == 1);
                let prim_idx = prim_idxs[leaf.first_prim_idx_idx];
                let aabb = &prim_aabbs[prim_idx];

                opt_bvh_idx_to_prim_idx.push(Some(prim_idx));
                BvhNode {
                    left_node_idx: u32::MAX,
                    right_node_idx_or_prim_idx: prim_index_ids[prim_idx] as u32,
                    min: aabb.min().coords.into(),
                    max: aabb.max().coords.into(),
                    luminance: prim_luminances[prim_idx],
                    parent_idx: leaf.parent_idx as u32,
                    ..Default::default()
                }
            }
            BuildBvhNodeKind::InternalNode(ref internal_node) => {
                opt_bvh_idx_to_prim_idx.push(None);
                BvhNode {
                    left_node_idx: internal_node.left_child_idx as u32,
                    right_node_idx_or_prim_idx: internal_node.right_child_idx as u32,
                    min: (node.aabb.min().coords - padding).into(),
                    max: (node.aabb.max().coords + padding).into(),
                    parent_idx: internal_node.parent_idx as u32,
                    ..Default::default()
                }
            }
        })
        .collect::<Vec<_>>();

    // compute luminance values for non-leaf nodes
    // the luminance of a node is the sum of the luminance of its children

    // the list is topologically sorted so we can just iterate over it in reverse order, and be sure that all the children of a node have already been processed
    for i in (0..opt_bvh.len()).rev() {
        if opt_bvh[i].left_node_idx != u32::MAX {
            // internal node
            let left_child_idx = opt_bvh[i].left_node_idx as usize;
            let right_child_idx = opt_bvh[i].right_node_idx_or_prim_idx as usize;

            // process children
            for child_idx in [left_child_idx, right_child_idx] {
                let child = &opt_bvh[child_idx].clone();
                opt_bvh[i].luminance += child.luminance;
            }
        }
    }

    if opt_bvh[0].left_node_idx == u32::MAX {
        // leaf node
        (opt_bvh, prim_aabbs[0], prim_luminances[0])
    } else {
        // not leaf node
        let aabb = Aabb::NonEmpty {
            min: opt_bvh[0].min.into(),
            max: opt_bvh[0].max.into(),
        };

        let luminance = opt_bvh[0].luminance;
        (opt_bvh, aabb, luminance)
    }
}

pub fn build_tl_bvh(
    // the transformation applied to each primitive
    prim_isometries: &[Isometry3<f32>],
    // the bounding box of each primitive
    prim_aabbs: &[Aabb],
    // how much power is in each primitive
    prim_luminances: &[f32],
    // instance id of each bl bvh
    prim_index_ids: &[u32],
) -> Vec<BvhNode> {
    let n_prims = prim_aabbs.len();
    assert_eq!(n_prims, prim_luminances.len());
    assert_eq!(n_prims, prim_index_ids.len());
    assert_eq!(n_prims, prim_isometries.len());

    let mut prim_idxs = (0..n_prims).collect::<Vec<_>>();

    let prim_aabbs = prim_aabbs
        .into_iter()
        .zip(prim_isometries.iter())
        .map(|(aabb, isometry)| aabb.transform(isometry))
        .collect::<Vec<_>>();

    let prim_centroids = prim_aabbs
        .iter()
        .map(|aabb| Point3::from((aabb.min().coords + aabb.max().coords) / 2.0))
        .collect::<Vec<_>>();

    let mut nodes = vec![];

    // create root node
    let root_node_idx = insert_blas_leaf_node(
        BuildBvhLeaf {
            parent_idx: u32::MAX as usize,
            first_prim_idx_idx: 0,
            prim_count: n_prims,
        },
        &mut nodes,
        &prim_idxs,
        &prim_aabbs,
    );

    // surface area metric
    fn cost_function(aabb1: &Aabb, aabb2: &Aabb, count1: usize, count2: usize) -> f32 {
        aabb1.area() * count1 as f32 + aabb2.area() * count2 as f32
    }

    subdivide(
        root_node_idx,
        &mut prim_idxs,
        &prim_aabbs,
        &prim_centroids,
        &mut nodes,
        &cost_function,
    );

    let padding = vector![0.0001, 0.0001, 0.0001];

    // nodes now contains a list of all the nodes in the blas.
    // however, it contains rust constructs and is not able to be passed to the shader
    // we now need to convert it into the finalized state that is optimized for gpu consumption
    let mut opt_bvh = nodes
        .into_iter()
        .map(|node| match node.kind {
            BuildBvhNodeKind::Leaf(ref leaf) => {
                assert!(leaf.prim_count == 1);
                let prim_idx = prim_idxs[leaf.first_prim_idx_idx];
                BvhNode {
                    left_node_idx: u32::MAX,
                    right_node_idx_or_prim_idx: prim_index_ids[prim_idx] as u32,
                    min: prim_aabbs[prim_idx].min().coords.into(),
                    max: prim_aabbs[prim_idx].max().coords.into(),
                    luminance: prim_luminances[prim_idx],
                    parent_idx: leaf.parent_idx as u32,
                }
            }
            BuildBvhNodeKind::InternalNode(ref internal_node) => BvhNode {
                left_node_idx: internal_node.left_child_idx as u32,
                right_node_idx_or_prim_idx: internal_node.right_child_idx as u32,
                min: (node.aabb.min().coords - padding).into(),
                max: (node.aabb.max().coords + padding).into(),
                parent_idx: internal_node.parent_idx as u32,
                ..Default::default()
            },
        })
        .collect::<Vec<_>>();

    // compute luminance values for non-leaf nodes
    // the luminance of a node is the sum of the luminance of its children
    // the luminance of a leaf node is the luminance of the primitive it contains

    // the list is topologically sorted so we can just iterate over it in reverse order, and be sure that all the children of a node have already been processed
    for i in (0..opt_bvh.len()).rev() {
        if opt_bvh[i].left_node_idx != u32::MAX {
            // internal node
            let left_child = opt_bvh[opt_bvh[i].left_node_idx as usize].clone();
            let right_child = opt_bvh[opt_bvh[i].right_node_idx_or_prim_idx as usize].clone();

            // process children
            for child in [left_child, right_child] {
                opt_bvh[i].luminance += child.luminance;
            }
        }
    }

    opt_bvh
}

// creates a visualization of the blas by turning it into a mesh
fn create_blas_visualization(blas_nodes: &Vec<BvhNode>) -> Vec<Vertex3D> {
    fn create_blas_visualization_inner(
        node_idx: usize,
        blas_nodes: &Vec<BvhNode>,
        vertexes: &mut Vec<Vertex3D>,
    ) {
        let node = &blas_nodes[node_idx];
        let loc = Point3::from((Vector3::from(node.min) + Vector3::from(node.max)) / 2.0);
        let dims = Vector3::from(node.max) - Vector3::from(node.min);
        vertexes.extend(utils::cuboid(loc, dims));

        match blas_nodes[node_idx].left_node_idx {
            u32::MAX => {}
            _ => {
                create_blas_visualization_inner(node.left_node_idx as usize, blas_nodes, vertexes);
                create_blas_visualization_inner(
                    node.right_node_idx_or_prim_idx as usize,
                    blas_nodes,
                    vertexes,
                );
            }
        }
    }

    let mut vertexes = vec![];
    create_blas_visualization_inner(0, blas_nodes, &mut vertexes);

    vertexes
}

// pub fn test_blas() -> Vec<Vertex3D> {
//     let mut prim_aabbs = vec![];
//     let mut prim_centroids = vec![];
//     let mut prim_luminances = vec![];
//     let mut prim_gl_ids = vec![];
//     for i in 0..100 {
//         // find a random point
//         let x = rand::random::<f32>() * 40.0 - 20.0;
//         let y = rand::random::<f32>() * 40.0 - 20.0;
//         let z = rand::random::<f32>() * 40.0 - 20.0;
//         let luminance = rand::random::<f32>() * 10.0;

//         let v0 = Point3::new(x, y, z);
//         let v1 = Point3::new(x, y + 0.1, z);
//         let v2 = Point3::new(x, y, z + 0.1);

//         prim_aabbs.push(Aabb::from_points(&[v0, v1, v2]));
//         prim_centroids.push(Point3::from((v0.coords + v1.coords + v2.coords) / 3.0));
//         prim_luminances.push(luminance);
//         prim_gl_ids.push(i as u32);
//     }

//     let nodes = build_bvh(
//         &prim_centroids,
//         &prim_aabbs,
//         &prim_luminances,
//         None,
//         &prim_gl_ids,
//     );

//     create_blas_visualization(&nodes)
// }
