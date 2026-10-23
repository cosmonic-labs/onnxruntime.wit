//! Classifies a generated image with SqueezeNet through the onnxruntime:onnxruntime interface.
//!
//! Usage: squeezenet [path to squeezenet model.onnx] [execution provider, e.g. webgpu or cpu]

wit_bindgen::generate!({
    world: "example:squeezenet/squeezenet",
    // The onnxruntime:onnxruntime package comes from the repository's top-level wit directory.
    path: ["../../wit", "wit"],
    generate_all,
});

use onnxruntime::onnxruntime::inference_session::{ExecutionProvider, Session, SessionOptions};
use onnxruntime::onnxruntime::types::{Tensor, TensorElementType};

fn main() -> std::process::ExitCode {
    let mut args = std::env::args().skip(1);
    let model = args.next().unwrap_or_else(|| "squeezenet/model.onnx".to_string());
    let provider = args.next().unwrap_or_else(|| "webgpu".to_string());
    match wit_bindgen::block_on(classify(&model, &provider)) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            std::process::ExitCode::FAILURE
        }
    }
}

async fn classify(model: &str, provider: &str) -> Result<(), String> {
    let options = SessionOptions {
        execution_providers: vec![ExecutionProvider { name: provider.to_string(), options: vec![] }],
        graph_optimization_level: None,
        execution_mode: None,
        intra_op_num_threads: None,
        free_dimension_overrides: vec![],
        log_id: None,
        log_severity_level: None,
        config_entries: vec![],
    };
    let session = Session::create_from_path(model.to_string(), Some(options))
        .await
        .map_err(|e| format!("{:?}: {}", e.code, e.message))?;

    let input = &session.get_input_metadata().await[0];
    let output = &session.get_output_metadata().await[0];
    println!("input `{}`, output `{}`", input.name, output.name);

    // A deterministic stand-in for an image: 1x3x224x224 floats in [0, 1).
    let dims = vec![1, 3, 224, 224];
    let pixels: Vec<f32> = (0..3 * 224 * 224).map(|i| ((i * 7919) % 255) as f32 / 255.0).collect();
    let data: Vec<u8> = pixels.iter().flat_map(|p| p.to_le_bytes()).collect();
    let image = Tensor::from_data(TensorElementType::Float32, dims, data)
        .await
        .map_err(|e| format!("{:?}: {}", e.code, e.message))?;

    let outputs = session
        .run(vec![(input.name.clone(), &image)], None, None)
        .await
        .map_err(|e| format!("{:?}: {}", e.code, e.message))?;

    let (_, scores) = &outputs[0];
    let bytes = scores.get_data().await.map_err(|e| format!("{:?}: {}", e.code, e.message))?;
    let scores: Vec<f32> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();

    let mut top: Vec<usize> = (0..scores.len()).collect();
    top.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
    for &class in &top[..5] {
        println!("class {class:4}: {:.4}", scores[class]);
    }
    Ok(())
}
