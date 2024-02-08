use crate::math::{EPSILON, Float3};
use crate::render_components::{AABB, BVH, BVHNode, Triangle};

struct Clipped
{
    pub vertices: u32,
    pub v: [Float3; 9],
    pub bounds: AABB
}

#[derive(Clone, Copy)]
pub struct BVHTriangle
{
    pub bounds: AABB,
    pub internal_triangle: Triangle
}

impl Clipped
{
    pub fn new(triangle: &BVHTriangle, bounding_box: &AABB) -> Self
    {
        let mut v = [Float3::zero(); 9];
        v[0] = triangle.internal_triangle.vertex0;
        v[1] = triangle.internal_triangle.vertex1;
        v[2] = triangle.internal_triangle.vertex2;
        let mut vertices: usize = 3;

        for a in 0..3
        {
            let mut ntmp = 0;
            let mut tmp = [Float3::zero(); 9];
            let mut c: Float3;
            let mut v0 = v[vertices - 1];
            let mut v1: Float3;
            let mut plane = bounding_box.minimal(a);
            let mut x = (v0.get_axis(a) - plane) >= 0.0;
            for i in 0..vertices
            {
                v1 = v[i];
                let d0 = v0.get_axis(a) - plane;
                let d1 = v1.get_axis(a) - plane;
                if x && d1 >= 0.0
                {
                    /* both in */
                    tmp[ntmp] = v1;
                    ntmp += 1;
                }
                else if !x && d1 > 0.0
                {
                    // coming in: emit C and d1
                    c = v0 + (d0 / (d0 - d1)) * (v1 - v0);
                    c.set_axis(a, plane);
                    tmp[ntmp] = c;
                    ntmp += 1;
                    tmp[ntmp] = v1;
                    ntmp += 1;
                    x = true;
                }
                else if x && d1 < 0.0 // going out: emit C
                {
                    c = v0 + (d0 / (d0 - d1)) * (v1 - v0);
                    c.set_axis(a, plane);
                    tmp[ntmp] = c;
                    ntmp += 1;
                    x = false;
                }

                v0 = v1;
            }
            vertices = 0;
            if ntmp < 3
            {
                break;
            }
            v0 = tmp[ntmp - 1];
            plane = bounding_box.maximal(a);
            x = (plane - v0.get_axis(a)) >= 0.0;
            for i in 0..ntmp
            {
                v1 = tmp[i];
                let d0 = plane - v0.get_axis(a);
                let d1 = plane - v1.get_axis(a);
                if x && d1 >= 0.0
                {
                    /* both in */
                    v[vertices] = v1;
                    vertices += 1;
                }
                else if !x && d1 > 0.0
                {
                    // coming in: emit C and d1
                    c = v0 + (d0 / (d0 - d1)) * (v1 - v0);
                    c.set_axis(a, plane);
                    v[vertices] = c;
                    vertices += 1;
                    v[vertices] = v1;
                    vertices += 1;
                    x = true;
                }
                else if x && d1 < 0.0
                {
                    // going out: emit C
                    c = v0 + (d0 / (d0 - d1)) * (v1 - v0);
                    c.set_axis(a, plane);
                    v[vertices] = c;
                    vertices += 1;
                    x = false;
                }

                v0 = v1;
            }
            if vertices < 3
            {
                // nothing or degenerate
                break;
            }
        }
        // calculate bounding box
        let mut bounds = AABB::from_empty();
        for i in 0..vertices
        {
            bounds.grow(&v[i]);
        }

        Clipped
        {
            vertices: vertices as u32,
            v,
            bounds
        }
    }
}

pub struct BVHConstructor
{
    pub triangles: Vec<BVHTriangle>,
    pub bvh_nodes: Vec<BVHNode>,
    pub triangle_idx: Vec<usize>,
    pub triangle_tmp: Vec<usize>,
    pub root_node_idx: usize,
    pub nodes_used: usize,
    pub triangle_ptr: usize,
    pub spatial_splits: usize,
    pub is_spatial: bool,
    pub bin_size: usize
}


impl BVHConstructor
{
    pub fn from_mesh(triangles: &Vec<Triangle>, bounds: &AABB, bin_size: usize) -> Self
    {
        let prim_count = triangles.len();
        let mut bvh_triangles: Vec<BVHTriangle> = Vec::with_capacity(prim_count);
        let mut triangle_idx: Vec<usize> = Vec::with_capacity(prim_count);

        let mut tri_idx = 0;
        for triangle in triangles
        {

            let mut bounds = AABB::from_bounds(&triangle.vertex0, &triangle.vertex0);
            bounds.grow(&triangle.vertex1);
            bounds.grow(&triangle.vertex2);

            bvh_triangles.push(BVHTriangle
            {
                bounds,
                internal_triangle: triangle.clone(),
            });
            triangle_idx.push(tri_idx as usize);
            tri_idx += 1;
        }

        // build bvh
        let mut bvh_nodes: Vec<BVHNode> = Vec::with_capacity(prim_count * 2);

        bvh_triangles.iter_mut().for_each(|triangle|
            {
                triangle.bounds = AABB::from_bounds(&triangle.internal_triangle.vertex0, &triangle.internal_triangle.vertex0);
                triangle.bounds.grow(&triangle.internal_triangle.vertex1);
                triangle.bounds.grow(&triangle.internal_triangle.vertex2);
            });

        let root = BVHNode {
            left_first: 0,
            tri_count: prim_count,
            bounds: *bounds
        };

        bvh_nodes.push(root);
        for _ in 1..prim_count * 2
        {
            bvh_nodes.push(
                BVHNode {
                    left_first: 0,
                    tri_count: 0,
                    bounds: AABB::from_empty()
                }
            );
        }

        let root_node_idx: usize = 0;

        let spatial_splits: usize = 0;

        let mut bvh = BVHConstructor
        {
            triangles: bvh_triangles,
            bvh_nodes,
            triangle_idx,
            triangle_tmp: Vec::new(),
            root_node_idx,
            nodes_used: 2,
            triangle_ptr: prim_count,
            spatial_splits,
            is_spatial: false,
            bin_size
        };

        bvh.subdivide(root_node_idx, 0);

        /* println!("SAH:            {}", bvh.get_total_sah(0));
         println!("nodes:          {}", bvh.get_node_count(0));
         println!("leafs:          {}", bvh.get_leaf_count(0));
         println!("spatial splits: {}", bvh.spatial_splits);
         println!("SAH cost:       {}", bvh.sah_cost(0));*/

        return bvh;
    }

    fn get_total_sah(&self, node_idx: usize) -> f32
    {
        let node = &self.bvh_nodes[node_idx];
        if node.is_leaf()
        {
            return node.bounds.area();
        }
        return self.get_total_sah(node.left_first) + self.get_total_sah(node.left_first + 1);
    }

    fn get_node_count(&self, node_idx: usize) -> u32
    {
        let node = &self.bvh_nodes[node_idx];
        if node.is_leaf()
        {
            return 1;
        }
        return self.get_node_count(node.left_first) + self.get_node_count(node.left_first + 1) + 1;
    }

    fn get_leaf_count(&self, node_idx: usize) -> u32
    {
        let node = &self.bvh_nodes[node_idx];
        if node.is_leaf()
        {
            return 1;
        }
        return self.get_leaf_count(node.left_first) + self.get_leaf_count(node.left_first + 1);
    }

    fn sah_cost(&self, node_idx: usize) -> f32
    {
        let node = &self.bvh_nodes[node_idx];
        let area = node.bounds.area();
        if node.is_leaf()
        {
            return area * (node.tri_count as f32);
        }
        let mut cost: f32 = 0.0;
        cost += self.sah_cost(node.left_first);
        cost += self.sah_cost(node.left_first + 1);
        cost += area;
        if node_idx == 0
        {
            cost *= 1.0 / area;
        }
        return cost;
    }

    fn finalize_sbvh(&mut self, node_idx: usize)
    {
        let node = self.bvh_nodes[node_idx];
        if node.is_leaf()
        {
            for i in 0..node.tri_count
            {
                let idx = self.triangle_idx[node.left_first + i];
                self.triangle_idx[node.left_first + i] = self.triangles[idx].internal_triangle.tri_idx as usize;
            }
            return;
        }
        self.finalize_sbvh(node.left_first);
        self.finalize_sbvh(node.left_first + 1);
    }

    fn find_best_object_split_plane(&mut self, node: &BVHNode, axis: &mut usize, split_pos: &mut f32, l_box: &mut AABB, r_box: &mut AABB) -> f32
    {
        let mut best_cost: f32 = 1e30;
        for a in 0..3
        {
            let mut bounds_min: f32 = 1e30;
            let mut bounds_max: f32 = -1e30;
            for i in 0..node.tri_count
            {
                let triangle = &self.triangles[self.triangle_idx[node.left_first + i]];

                let triangle_center = triangle.bounds.center( a );
                bounds_min = bounds_min.min( triangle_center );
                bounds_max = bounds_max.max( triangle_center );
            }
            if bounds_min == bounds_max
            {
                continue;
            }
            // loop over split plane candidates
            let bin_width = (bounds_max - bounds_min) / (self.bin_size as f32);
            for b in 1..self.bin_size
            {
                let plane = bounds_min + bin_width * (b as f32);

                let mut left_box = AABB::from_empty();
                let mut right_box = AABB::from_empty();

                let mut left_count: u32 = 0;
                let mut right_count: u32 = 0;
                for i in 0..node.tri_count
                {
                    let triangle = &self.triangles[self.triangle_idx[node.left_first + i]];
                    if triangle.bounds.center( a ) < plane
                    {
                        left_box.grow_aabb( &triangle.bounds );
                        left_count += 1;
                    }
                    else
                    {
                        right_box.grow_aabb( &triangle.bounds );
                        right_count += 1;
                    }
                }
                let plane_cost = (left_count as f32) * left_box.area() + (right_count as f32) * right_box.area();
                if plane_cost < best_cost
                {
                    *axis = a;
                    *split_pos = plane;
                    best_cost = plane_cost;
                    *l_box = left_box;
                    *r_box = right_box;
                }

            }
        }

        return best_cost;
    }

    fn find_best_spatial_split_plane(&mut self, node: &BVHNode, axis: &mut usize, split_pos: &mut f32, n_left: &mut usize, n_right: &mut usize, bounds_left: &mut AABB, bounds_right: &mut AABB, splitted: &mut i32) -> f32
    {
        let mut best_cost = 1e30;
        for a in 0..3
        {
            let bounds_min = node.bounds.min_bound.get_axis(a);
            let bounds_max = node.bounds.max_bound.get_axis(a);
            if bounds_min == bounds_max
            {
                continue;
            }

            // calculate cost of spatial splits
            let bin_extend = (bounds_max - bounds_min) / (self.bin_size as f32);
            for b in 1..self.bin_size
            {
                // calculate spatial split plane position
                let pos = bounds_min + (b as f32) * bin_extend;
                // construct left and right bounding box
                let mut left_bounds = AABB::from_empty();
                let mut right_bounds = AABB::from_empty();
                let mut left_box = AABB::from_bounds(&node.bounds.min_bound, &node.bounds.max_bound);
                let mut right_box = AABB::from_bounds(&node.bounds.min_bound, &node.bounds.max_bound);

                left_box.max_bound.set_axis(a, pos - EPSILON);
                right_box.min_bound.set_axis(a, pos + EPSILON);

                let mut left_count: usize = 0;
                let mut right_count: usize = 0;
                // loop over triangles
                for i in 0..node.tri_count
                {
                    // fetch a triangle
                    let triangle = &self.triangles[self.triangle_idx[node.left_first + i]];

                    // extend left and right side
                    let left_triangle = Clipped::new(&triangle, &left_box);

                    if left_triangle.vertices > 2
                    {
                        left_bounds.grow_aabb( &left_triangle.bounds );
                        left_count += 1;
                    }

                    let right_triangle = Clipped::new(&triangle, &right_box);

                    if right_triangle.vertices > 2
                    {
                        right_bounds.grow_aabb( &right_triangle.bounds );
                        right_count += 1;
                    }
                }
                // calculate cost for this split plane
                if left_count > 0 && right_count > 0
                {
                    let zero: f32 = 0.0;
                    let cost = zero.max(left_bounds.area()) * (left_count as f32) + zero.max(right_bounds.area()) * (right_count as f32);

                    if cost < best_cost
                    {
                        *axis = a;
                        best_cost = cost;
                        *split_pos = pos;

                        *splitted = ((right_count + left_count) as i32) - (node.tri_count as i32);
                        *n_left = left_count;
                        *n_right = right_count;
                        *bounds_left = left_bounds;
                        *bounds_right = right_bounds;
                    }

                }
            }
        }
        return best_cost;
    }

    fn calculate_node_cost(node: &BVHNode) -> f32
    {
        let e = node.bounds.max_bound - node.bounds.min_bound;
        let area = e.x * e.y + e.y * e.z + e.z * e.x;
        return (node.tri_count as f32) * area;
    }

    fn subdivide(&mut self, node_idx: usize, slack: usize)
    {
        let node = self.bvh_nodes[node_idx].clone();
        let mut obj_split_axis: usize = 0;
        let mut spatial_split_axis: usize = 0;
        let mut splitted: i32 = 0;
        let mut left_box = AABB::from_empty();
        let mut right_box = AABB::from_empty();

        let mut obj_split_pos: f32 = 0.0;
        let mut spatial_split_pos: f32 = 0.0;
        let no_split_cost: f32 = BVHConstructor::calculate_node_cost(&node);

        let obj_split_cost = self.find_best_object_split_plane(&node, &mut obj_split_axis, &mut obj_split_pos, &mut left_box, &mut right_box);

        if self.is_spatial
        {
            let root_area = self.bvh_nodes[0].bounds.area();
            let lambda = left_box.intersection( &right_box ).area() / root_area;

            if lambda > 1e-5
            {
                let mut n_left: usize = 0;
                let mut n_right: usize = 0;
                let mut bounds_left = AABB::from_empty();
                let mut bounds_right = AABB::from_empty();

                let mut spatial_split_cost = self.find_best_spatial_split_plane(&node, &mut spatial_split_axis, &mut spatial_split_pos, &mut n_left, &mut n_right, &mut bounds_left, &mut bounds_right, &mut splitted);

                let mut n_left: i32 = n_left as i32;
                let mut n_right: i32 = n_right as i32;

                if spatial_split_cost < obj_split_cost && splitted < (slack as i32)
                {
                    if spatial_split_cost >= no_split_cost
                    {
                        return; // don't split, not worth it
                    }
                    let mut left_of_split = node.bounds.clone();
                    let mut right_of_split = node.bounds.clone();
                    left_of_split.max_bound.set_axis(spatial_split_axis, spatial_split_pos + EPSILON);
                    right_of_split.min_bound.set_axis(spatial_split_axis, spatial_split_pos - EPSILON);

                    let mut left_pos = node.left_first;
                    let mut left_count: usize = 0;
                    let mut right_pos = node.left_first + node.tri_count + slack;
                    let mut right_count: usize = 0;

                    for i in 0..node.tri_count
                    {
                        let mut idx = self.triangle_idx[node.left_first + i];
                        let mut left_part = Clipped::new(&self.triangles[idx], &left_of_split);
                        let mut right_part = Clipped::new(&self.triangles[idx], &right_of_split);
                        let mut in_left = left_part.vertices >= 3;
                        let mut in_right = right_part.vertices >= 3;

                        if in_left && in_right
                        {
                            let c1 = bounds_left.union(&self.triangles[idx].bounds).area() * (n_left as f32) + bounds_right.area() * ((n_right - 1) as f32);
                            let c2 = bounds_left.area() * ((n_left - 1) as f32) + bounds_right.union( &self.triangles[idx].bounds ).area() * (n_right as f32);
                            if c1 < spatial_split_cost || c2 < spatial_split_cost
                            {
                                if c1 < c2
                                {
                                    spatial_split_cost = c1;
                                    n_right -= 1;
                                    bounds_left.grow_aabb( &self.triangles[idx].bounds );
                                    left_part.bounds = self.triangles[idx].bounds;
                                    in_right = false; // undo clip
                                }
                                else {
                                    spatial_split_cost = c2;
                                    n_left -= 1;
                                    bounds_right.grow_aabb( &self.triangles[idx].bounds );
                                    right_part.bounds = self.triangles[idx].bounds;
                                    in_left = false; // undo clip
                                }
                            }
                        }

                        if in_left
                        {
                            self.triangle_idx[left_pos] = idx;
                            left_pos += 1;
                            self.triangles[idx].bounds = left_part.bounds;
                            left_count += 1;
                        }
                        if in_right
                        {
                            if in_left
                            {
                                self.triangles[self.triangle_ptr] = self.triangles[idx].clone();
                                idx = self.triangle_ptr;
                                self.triangle_ptr += 1;
                            }
                            right_pos -= 1;
                            self.triangle_tmp[right_pos] = idx;
                            self.triangles[idx].bounds = right_part.bounds;
                            right_count += 1;
                        }
                    }

                    let slack = ((slack as i32) - splitted) as usize;
                    let half_slack = slack / 2;

                    let triangle_idx_adr = node.left_first + left_count + half_slack;
                    let triangle_tmp_adr = right_pos;

                    for i in 0..right_count
                    {
                        self.triangle_idx[triangle_idx_adr + i] = self.triangle_tmp[triangle_tmp_adr + i];
                    }

                    let left_child_idx = self.nodes_used;
                    self.nodes_used += 1;
                    let right_child_idx = self.nodes_used;
                    self.nodes_used += 1;

                    self.bvh_nodes[left_child_idx].left_first = node.left_first;
                    self.bvh_nodes[left_child_idx].tri_count = left_count;
                    self.bvh_nodes[right_child_idx].left_first = node.left_first + left_count + half_slack;
                    self.bvh_nodes[right_child_idx].tri_count = right_count;

                    self.bvh_nodes[node_idx].left_first = left_child_idx;
                    self.bvh_nodes[node_idx].tri_count = 0;

                    self.bvh_nodes[left_child_idx].bounds = bounds_left;
                    self.bvh_nodes[right_child_idx].bounds = bounds_right;
                    self.spatial_splits += 1;
                    self.subdivide(left_child_idx, half_slack);
                    self.subdivide(right_child_idx, half_slack);
                    return;
                }
            }
        }

        if obj_split_cost >= no_split_cost
        {
            return;
        }

        let mut i = node.left_first;
        let mut j: i32 = ((i + node.tri_count) as i32) - 1;

        while (i as i32) <= j
        {
            let jx = j as usize;
            let triangle = &self.triangles[self.triangle_idx[i]];
            if triangle.bounds.center(obj_split_axis) < obj_split_pos
            {
                i += 1;
            }
            else {
                if i != jx
                {
                    self.triangle_idx.swap(i, jx);
                }
                j -= 1;
            }
        }

        let half_slack = slack / 2;
        let left_count = i - node.left_first;
        if left_count == 0 || left_count == node.tri_count
        {
            return;
        }

        if half_slack > 0
        {
            let lb = i;
            let ub = i + (node.tri_count - left_count);
            self.triangle_idx.copy_within(lb..ub, i + half_slack);
        }

        let left_child_idx = self.nodes_used;
        self.nodes_used += 1;
        let right_child_idx = self.nodes_used;
        self.nodes_used += 1;
        self.bvh_nodes[left_child_idx].left_first = node.left_first;
        self.bvh_nodes[left_child_idx].tri_count = left_count;
        self.bvh_nodes[right_child_idx].left_first = i + half_slack;
        self.bvh_nodes[right_child_idx].tri_count = node.tri_count - left_count;

        self.bvh_nodes[node_idx].left_first = left_child_idx;
        self.bvh_nodes[node_idx].tri_count = 0;

        self.bvh_nodes[left_child_idx].bounds = left_box;
        self.bvh_nodes[right_child_idx].bounds = right_box;
        self.subdivide(left_child_idx, half_slack);
        self.subdivide(right_child_idx, half_slack);
    }
}