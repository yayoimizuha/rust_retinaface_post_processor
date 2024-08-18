use std::f32;
use std::ops::{Div, Mul};
use std::slice::from_raw_parts;
use anyhow::Result;
use itertools::{enumerate, iproduct};
use ndarray::{arr2, Array, Axis, concatenate, Ix2, s, Array3};
use powerboxesrs::nms::nms;
use cached::proc_macro::cached;

#[cached(size = 100)]
fn prior_box(min_sizes: Vec<Vec<usize>>, steps: Vec<usize>, clip: bool, image_size: [usize; 2]) -> (Array<f32, Ix2>, usize) {
    let feature_maps = steps.iter().map(|&step| {
        [f32::ceil(image_size[0] as f32 / step as f32) as i32,
            f32::ceil(image_size[1] as f32 / step as f32) as i32]
    }).collect::<Vec<_>>();
    let mut anchors: Vec<[f32; 4]> = vec![];
    for (k, f) in enumerate(feature_maps) {
        for (i, j) in iproduct!(0..f[0],0..f[1]) {
            let t_min_sizes = &min_sizes[k];
            for &min_size in t_min_sizes {
                let s_kx = min_size as f32 / image_size[0] as f32;
                let s_ky = min_size as f32 / image_size[1] as f32;
                let dense_cx = [j as f32 + 0.5].iter().map(|x| x * steps[k] as f32 / image_size[0] as f32).collect::<Vec<_>>();
                let dense_cy = [i as f32 + 0.5].iter().map(|y| y * steps[k] as f32 / image_size[1] as f32).collect::<Vec<_>>();
                for (cy, cx) in iproduct!(dense_cy,dense_cx) {
                    anchors.push([cx, cy, s_kx, s_ky]);
                }
            }
        }
    }
    let mut output = arr2(&anchors);
    if clip {
        output = output.mapv(|x| f32::min(f32::max(x, 0.0), 1.0));
    }
    (output, anchors.len())
}

fn decode(loc: Array<f32, Ix2>, priors: Array<f32, Ix2>, variances: [f32; 2]) -> Array<f32, Ix2> {
    let mut boxes = concatenate(Axis(1), &*vec![
        (priors.slice(s![..,..2]).to_owned() + loc.slice(s![..,..2]).mul(variances[0]) * priors.slice(s![..,2..])).view(),
        (priors.slice(s![..,2..]).to_owned() * loc.slice(s![..,2..]).mul(variances[1]).to_owned().mapv(f32::exp)).view(),
    ]).unwrap();


    let boxes_sub = boxes.slice(s![..,..2]).to_owned() - boxes.slice(s![..,2..]).div(2.0);
    boxes.slice_mut(s![..,..2]).assign(&boxes_sub);

    let boxes_add = boxes.slice(s![..,2..]).to_owned() + boxes.slice(s![..,..2]);
    boxes.slice_mut(s![..,2..]).assign(&boxes_add);
    boxes
}


fn decode_landmark(pre: Array<f32, Ix2>, priors: Array<f32, Ix2>, variances: [f32; 2]) -> Array<f32, Ix2> {
    concatenate(Axis(1),
                &*vec![
                    (priors.slice(s![..,..2]).to_owned() + pre.slice(s![..,..2]).mapv(|x| x * variances[0]) * priors.slice(s![..,2..])).view(),
                    (priors.slice(s![..,..2]).to_owned() + pre.slice(s![..,2..4]).mapv(|x| x * variances[0]) * priors.slice(s![..,2..])).view(),
                    (priors.slice(s![..,..2]).to_owned() + pre.slice(s![..,4..6]).mapv(|x| x * variances[0]) * priors.slice(s![..,2..])).view(),
                    (priors.slice(s![..,..2]).to_owned() + pre.slice(s![..,6..8]).mapv(|x| x * variances[0]) * priors.slice(s![..,2..])).view(),
                    (priors.slice(s![..,..2]).to_owned() + pre.slice(s![..,8..10]).mapv(|x| x * variances[0]) * priors.slice(s![..,2..])).view(),
                ]).unwrap()
}

pub fn infer(data: [*const f32; 3], batch_size: usize, image_size: [usize; 2]) -> Result<Vec<Vec<Vec<Vec<f32>>>>> {
    let confidence_threshold = 0.7;
    let nms_threshold = 0.4;
    let variance = [0.1, 0.2];

    let transformed_size = Array::from_iter(image_size).into_owned();

    let (prior_box, onnx_output_width) = prior_box(
        vec![vec![16, 32], vec![64, 128], vec![256, 512]],
        [8, 16, 32].into(),
        false,
        [image_size[0], image_size[1]],
    );

    let extract = |tensor: *const f32, w: usize| Array3::from_shape_vec(
        (batch_size, onnx_output_width, w), unsafe {
            from_raw_parts(tensor, batch_size * onnx_output_width * w).to_vec()
        },
    ).unwrap().to_owned();
    let [ _landmark, _confidence, _loc] = [(10, data[0]), (2, data[1]), (4, data[2])].map(|(w, dat)| extract(dat, w));
    let mut all_faces = vec![];
    for i in 0..batch_size {
        let landmark = _landmark.slice(s![i,..,..]).insert_axis(Axis(0)).to_owned();
        let confidence = _confidence.slice(s![i,..,..]).insert_axis(Axis(0)).to_owned();
        let loc = _loc.slice(s![i,..,..]).insert_axis(Axis(0)).to_owned();

        let scale_landmarks = concatenate(Axis(0), &*vec![transformed_size.view(); 5])?.mapv(|x| x as f32);
        let scale_bboxes = concatenate(Axis(0), &*vec![transformed_size.view(); 2])?.mapv(|x| x as f32);

        let confidence_exp = confidence.map(|v| v.exp());
        let confidence = confidence_exp.clone() / confidence_exp.sum_axis(Axis(2)).insert_axis(Axis(2));

        let mut boxes = decode(loc.slice(s![0,..,..]).to_owned(), prior_box.clone(), variance);
        boxes = boxes * scale_bboxes;

        let mut scores = confidence.slice(s![0,..,1]).to_owned();

        let mut landmarks = decode_landmark(landmark.slice(s![0,..,..]).to_owned(), prior_box.clone(), variance);
        landmarks = landmarks * scale_landmarks;


        let valid_index = scores.iter().enumerate().filter(|(_, val)| val > &&confidence_threshold).map(|(order, _)| order).collect::<Vec<_>>();
        boxes = boxes.select(Axis(0), &*valid_index);
        landmarks = landmarks.select(Axis(0), &*valid_index);
        scores = scores.select(Axis(0), &*valid_index);


        let keep = nms(&boxes.to_owned(), &scores.mapv(|x| x as f64).to_owned(), nms_threshold, confidence_threshold as f64);


        let mut faces = vec![];
        for index in keep {
            faces.push(vec![boxes.slice(s![index,..]).to_vec(),
                            vec![*scores.get(index).unwrap()],
                            landmarks.slice(s![index,..]).to_vec()
            ]);
        }
        all_faces.push(faces);
    }
    Ok(all_faces)
}
