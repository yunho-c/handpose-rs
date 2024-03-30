#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_must_use)]

// basics
use std::error::Error;
use std::iter::Filter;
use std::path::{Path, PathBuf};
// arrays/vectors/tensors
use ndarray::{array, s, Array, Array1, Array2, Array3, Array4, Axis, ArrayBase};
use ndarray::{OwnedRepr, Dim, IxDyn, Ix2};
use ort::{GraphOptimizationLevel, Session};
// images
use image::io::Reader as ImageReader;
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb, Rgba};
use image::imageops::FilterType;
use imageproc::drawing::draw_filled_rect_mut;
use imageproc::rect::Rect;
// use show_image::{event, AsImageView, WindowOptions};
mod calculate;
mod visualize;

struct HandResult {
  palm: PalmResult,
  landmark: LandmarkResult,
}

struct PalmResult {
  cx: f32,
  cy: f32,
  size: f32, 
  rotation: f32,
  // score: f32,
  handedness: String,
  imshape: (u32, u32),
}

impl PalmResult {
  pub fn abs(self) -> (u32, u32, u32) {
    return ((self.cx * self.imshape.0 as f32) as u32, 
            (self.cy * self.imshape.1 as f32) as u32, 
            (self.size * std::cmp::max(self.imshape.0, self.imshape.1) as f32) as u32);
  }
}

struct LandmarkResult {
  coords: Array3<f32>, // 21x3
}

struct Handpose {
  palm_file: PathBuf,
  palm_model: Session,
  palm_size: u32,

  ldmk_file: PathBuf,
  ldmk_model: Session,
  ldmk_size: u32,

  detection_threshold: f32
}

impl Handpose {
  pub fn new() -> Result<Self, Box<dyn Error>>{
    // user params
    let detection_threshold = 0.5;
    
    // models
    let palm_file = PathBuf::from("./hand-gesture-recognition-using-onnx/model/palm_detection/palm_detection_full_inf_post_192x192.onnx");
    let palm_model = Session::builder()?
      .with_optimization_level(GraphOptimizationLevel::Level3)?
      .with_intra_threads(4)?
      .with_model_from_file(&palm_file)?;
    let palm_size = 192;

    let ldmk_file = PathBuf::from("./hand-gesture-recognition-using-onnx/model/hand_landmark/hand_landmark_sparse_Nx3x224x224.onnx");
    let ldmk_model = Session::builder()?
      .with_optimization_level(GraphOptimizationLevel::Level3)?
      .with_intra_threads(4)?
      .with_model_from_file(&ldmk_file)?;
    let ldmk_size = 224;

    Ok( Self { palm_file, palm_model, palm_size, ldmk_file, ldmk_model, ldmk_size, detection_threshold })
  }

  /// Processes image and returns detected hands (size, location, orientation, landmark points).
  /// ### Notes on variable naming:
  /// - cx, cy -> center point's x-coordinate & y-coordinate
  /// - "abs" means it's measured in terms of pixels
  /// 
  /// ### Variable Details
  /// palm_preds: [confidence, cx, cy, boxsize, kp0x, kp0y, kp2x, kp2y]
  /// palm_dets:  [size (rel), rotation (rad), cx (rel), cy (rel)]
  ///   TODO palm_dets:  [cx (rel), cy (rel), size (rel), rotation (rad), score] 
  /// palm_rects: [cx (abs), cy (abs), box_width (abs), box_height (abs), rotation (deg)] 
  ///   TODO palm_rects: [cx (abs), cy (abs), size (abs), rotation (deg), score] 
  /// TODO perhaps rename things to be better representative. 
  /// pub fn process(self, image: DynamicImage) -> Result<(), Box<dyn Error>> {
  /// pub fn process(self, image: DynamicImage) -> Result<HandResult, Box<dyn Error>> {
  ///
  /// ### Returns
  /// return format: a list of structs. 
  /// palm: contains palm info, in both abs (TODO) & rel: n * [cx, cy, size, score, orientation, handedness]
  /// landmark: contains ldmk info, in both abs & rel: [n*21*3]
  /// perhaps even provide an appearance embedding (provided by segmentation mask or hand-specific embedder) & gesture info!

  pub fn process(self, image: DynamicImage) -> Result<Vec<HandResult>, Box<dyn Error>> {
    let (original_width, original_height) = image.dimensions();
    // Palm Detection
    let palm_img = image.resize_exact(self.palm_size, self.palm_size, FilterType::CatmullRom);
    let palm_inputs = ort::inputs!["input" => self.image_to_onnx_input(palm_img.clone()).view()]?;
    // println!("{:?}", palm_inputs);
    let palm_outputs = self.palm_model.run(palm_inputs)?;
    let palm_preds = palm_outputs["pdscore_boxx_boxy_boxsize_kp0x_kp0y_kp2x_kp2y"].extract_tensor::<f32>()?;
    let palm_preds_view = palm_preds.view().t().slice(s![.., ..]).into_owned();
    let palm_imshape = (self.palm_size as _, self.palm_size as _);
    let palm_dets = calculate::postprocess_palms(palm_imshape, palm_preds_view, self.detection_threshold);
    // print!("palm_dets: {:?}", palm_dets);
    let palm_rects = calculate::calculate_rects(palm_dets, palm_imshape);
    // print!("palm_rects: {:?}", rects);
    let det_size = palm_rects.get((3, 0)).unwrap();
    // println!("det_size: {:?}", det_size);
    let rot_crops = calculate::rotate_and_crop_rectangle(&image, palm_rects.clone());
    // TODO make this come from the original image and not the palm_img (Nyquist)
    // print!("{:?}", rot_crops);
    let hand_count = rot_crops.len(); // TODO find a better implementation ?
    let mut ldmk_imgs: Vec<DynamicImage> = Vec::new();
    for i in 0..hand_count {
      let ldmk_img = rot_crops[i].resize_exact(self.ldmk_size, self.ldmk_size, FilterType::CatmullRom);
      ldmk_imgs.push(ldmk_img)
    }
    // Hand Landmark Detection
    let ldmk_inputs = ort::inputs!["input" => self.images_to_onnx_input(ldmk_imgs).view()]?;
    // println!("{:?}", ldmk_inputs);
    let ldmk_outputs = self.ldmk_model.run(ldmk_inputs)?;
    let ldmk_preds = ldmk_outputs["xyz_x21"].extract_tensor::<f32>()?;
    let ldmk_preds_view = ldmk_preds.view().clone().into_owned();
    // println!("ldmk_preds_view: {:?}", ldmk_preds_view);
    let ldmk_rels = (ldmk_preds_view / 224.).clone().into_owned().into_shape((hand_count, 21, 3))?;
    // println!("ldmk_rels: {:?}", ldmk_rels);

    let results: Vec<HandResult> = Vec::new();
    for i in 0..hand_count {
      let palm = PalmResult {
        cx: palm_dets[[i, 2]],
        cy: palm_dets[[i, 3]],
`       size: palm_dets[[i, 0]],
        rotation: palm_dets[[i, 1]],
        // score: 
        handedness: "Unknown".to_string(),
        imshape: (original_width, original_height),
      };
      let ldmk = LandmarkResult {
        coords: ldmk_rels,
      };
    }

    Ok(results)
  }

  fn image_to_onnx_input(&self, image: DynamicImage) -> Array4<f32> { // HWC -> NCHW
    let (width, height) = image.dimensions();
    let channels = 3;
    let mut onnx_input = Array::zeros((1, channels, height as _, width as _));
    for (x, y, pixel) in image.into_rgb8().enumerate_pixels() {
        let (r, g, b) = (pixel[0], pixel[1], pixel[2]);
        // Set the RGB values in the array
        onnx_input[[0, 0, y as _, x as _]] = (r as f32) / 255.;
        onnx_input[[0, 1, y as _, x as _]] = (g as f32) / 255.;
        onnx_input[[0, 2, y as _, x as _]] = (b as f32) / 255.;
      };
    onnx_input
  }

  fn images_to_onnx_input(&self, images: Vec<DynamicImage>) -> Array4<f32> {
    for i in 0..images.len() {
      let mut img_arr = images[i].to_rgb8().into_vec();
      let (width, height) = images[i].dimensions();
      let channels = 3;
      let mut onnx_input = Array::zeros((images.len(), channels, height as _, width as _));
      for (x, y, pixel) in images[i].clone().into_rgb8().enumerate_pixels() {
        let (r, g, b) = (pixel[0], pixel[1], pixel[2]);
        onnx_input[[i, 0, y as _, x as _]] = (r as f32) / 255.;
        onnx_input[[i, 1, y as _, x as _]] = (g as f32) / 255.;
        onnx_input[[i, 2, y as _, x as _]] = (b as f32) / 255.;
      }
    return onnx_input;
    }
    return Array::zeros((1, 3, 224, 224));
  } 
}

/// Test functionality
fn test() -> Result<(), Box<dyn Error>> {
  // User Params
  // Set test image
  const TEST_NAME: &str = "hand";
  // const TEST_NAME: &str = "hand_90";
  // const TEST_NAME: &str = "hand_180";
  // const TEST_NAME: &str = "hand_270";
  // const TEST_NAME: &str = "bottom_left";
  // const TEST_NAME: &str = "top_right";

  let test_img_path = format!("./tests/test_images/{}.jpg", TEST_NAME);
  let image = image::open(test_img_path)?;

  let hp = Handpose::new()?;
  
  let hand_results = hp.process(image)?;
  
  // let scale_factor = 1.;
  visualize::visualize(image, palm_rects.slice(s![.., 0]).into_owned(), ldmk_preds_rel, scale_factor); // ALT1

  Ok(())
}


fn main() -> Result<(), Box<dyn Error>>{
  test()?;
  Ok(())
}

// NOTE let's develop post-processing / image manipulation step
// not by using ONNX but just with test data and tracing & modeling (understanding) of Python script
