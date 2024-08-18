mod retinaface_resnet;
mod found_face;
use pyo3::prelude::*;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pyfunction]
fn resnet_post_process(ptr: Vec<usize>, batch_size: usize, image_size: Vec<usize>) -> Vec<Vec<Vec<Vec<f32>>>> {
    let ptr_arr: [*const f32; 3];
    match ptr.len() {
        3 => {
            ptr_arr = ptr.into_iter().map(|x| x as *const f32).collect::<Vec<_>>().try_into().unwrap();
        }
        _ => {
            eprintln!("need 3 items.");
            panic!()
        }
    }
    let image_size_arr: [usize; 2];
    match image_size.len() {
        2 => {
            image_size_arr = image_size.try_into().unwrap();
        }
        _ => {
            eprintln!("need 3 items.");
            panic!()
        }
    }
    let a = retinaface_resnet::infer(ptr_arr, batch_size, image_size_arr).unwrap();
    // println!("{:?}", a);
    a
}


/// A Python module implemented in Rust.
#[pymodule]
fn rust_retinaface_post_processor(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(resnet_post_process, m)?)?;
    Ok(())
}
