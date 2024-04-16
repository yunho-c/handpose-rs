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
use ndarray::{array, s, Array, Array1, Array2, Array3, Array4, ArrayBase, ArrayD, Axis, IxDynImpl};
use ndarray::{OwnedRepr, Dim, IxDyn, Ix2};
use ort::{GraphOptimizationLevel, InMemorySession, InMemorySession, Session};
// images
use image::io::Reader as ImageReader;
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb, Rgba};
use image::imageops::FilterType::{self, CatmullRom};
use imageproc::drawing::draw_filled_rect_mut;
use imageproc::rect::Rect;
// use show_image::{event, AsImageView, WindowOptions};
pub mod calculate;
pub mod visualize;

use serde::{Serialize, Deserialize};

static PALM_MODEL_BYTES: &[u8] = include_bytes!("../model/palm_detection_full_inf_post_192x192.ort");
static LDMK_MODEL_BYTES: &[u8] = include_bytes!("../model/hand_landmark_sparse_Nx3x224x224.ort");

#[derive(Debug)]
pub struct HandResult {
  pub palm: PalmResult,
  pub landmark: LandmarkResult,
}

#[derive(Debug)]
pub struct PalmResult {
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


#[derive(Debug)]
pub struct LandmarkResult {
  pub coords: Array2<f32>, // 21x3
}

pub struct Handpose {
  palm_file: PathBuf,
  palm_model: InMemorySession<'static>,
  palm_size: u32,

  ldmk_file: PathBuf,
  ldmk_model: InMemorySession<'static>,
  ldmk_size: u32,

  detection_threshold: f32
}

static PALM_FILE: &[u8] = include_bytes!("../hand-gesture-recognition-using-onnx/model/palm_detection/palm_detection_full_inf_post_192x192.ort");
static LANDMARK_FILE: &[u8] = include_bytes!("../hand-gesture-recognition-using-onnx/model/hand_landmark/hand_landmark_sparse_Nx3x224x224.ort");

impl Handpose {
  pub fn new() -> Result<Self, Box<dyn Error>>{
    #[cfg(target_arch = "wasm32")]
    ort::wasm::initialize();

    // user params
    let detection_threshold = 0.5;
    
    // models
    let palm_model = Session::builder()?
      // .with_optimization_level(GraphOptimizationLevel::Level3)?
      .with_intra_threads(4)?
      .commit_from_memory_directly(PALM_MODEL_BYTES)?;
    let palm_size = 192;

    let ldmk_model = Session::builder()?
      // .with_optimization_level(GraphOptimizationLevel::Level3)?
      .with_intra_threads(4)?
      .commit_from_memory_directly(LDMK_MODEL_BYTES)?;
    let ldmk_size = 224;

    Ok( Self { palm_model, palm_size, ldmk_model, ldmk_size, detection_threshold })
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

  pub fn process(&&self, image: DynamicImage) -> Result<Vec<HandResult>, Box<dyn Error>> {
  // pub fn process(self, image: DynamicImage) -> Result<(), Box<dyn Error>> {
    let (original_width, original_height) = image.dimensions();
    // Palm Detection
    let palm_img = image.resize_exact(self.palm_size, self.palm_size, FilterType::CatmullRom);
    let palm_inputs = ort::inputs!["input" => self.image_to_onnx_input(palm_img.clone()).view()]?;
    // println!("{:?}", palm_inputs);
    let palm_outputs = self.palm_model.run(palm_inputs)?;
    let palm_preds = palm_outputs["pdscore_boxx_boxy_boxsize_kp0x_kp0y_kp2x_kp2y"].try_try_extract_tensor::<f32>()?;
    let palm_preds_view = palm_preds.view().t().slice(s![.., ..]).into_owned();
    let palm_imshape = (self.palm_size as _, self.palm_size as _);
    let palm_dets = calculate::postprocess_palms(palm_imshape, palm_preds_view, self.detection_threshold);
    // println!("palm_dets: {:?}", palm_dets);
    let palm_rects = calculate::calculate_rects(palm_dets.clone(), palm_imshape);
    // println!("palm_rects: {:?}", rects);
    let det_size = palm_rects.get((3, 0)).unwrap();
    // println!("det_size: {:?}", det_size);
    let rot_crops = calculate::rotate_and_crop_rectangle(&palm_img, palm_rects.clone()); // TODO make this come from the original image and not the palm_img (Nyquist)
    // println!("{:?}", rot_crops);
    let hand_count = rot_crops.len(); // TODO find a better implementation ?
    let mut ldmk_imgs: Vec<DynamicImage> = Vec::new();
    for i in 0..hand_count {
      let ldmk_img = rot_crops[i].resize_exact(self.ldmk_size, self.ldmk_size, FilterType::CatmullRom);
      ldmk_imgs.push(ldmk_img)
      // ldmk_img.clone().save("./ldmk_img.jpg"); // DEBUG
    }
    // Hand Landmark Detection
    let ldmk_inputs = ort::inputs!["input" => self.images_to_onnx_input(ldmk_imgs.clone()).view()]?;
    // println!("{:?}", ldmk_inputs);
    let ldmk_outputs = self.ldmk_model.run(ldmk_inputs)?;
    let ldmk_preds = ldmk_outputs["xyz_x21"].try_extract_tensor::<f32>()?;
    let ldmk_preds_view = ldmk_preds.view().clone().into_owned();
    // println!("ldmk_preds_view: {:?}", ldmk_preds_view);
    let ldmk_rels_local = (ldmk_preds_view / 224.).clone().into_owned().into_shape((hand_count, 21, 3))?;
    // println!("ldmk_rels: {:?}", ldmk_rels);


    let pixmap = visualize::visualize2(ldmk_imgs[0].clone(), ldmk_rels_local.slice(s![0, .., ..]).clone().into_owned());
    pixmap.save_png("./viz/ldmk_viz.png")?;

    // let ldmk_rels_global = 

    // computation essentially consists of: rotation, scaling, and translation. 
    // first, rotate around the center point. 
    // then, scale by the size of palm.
    // finally, translate so that the center point of the landmark is the center point of the palm box. 

    // let (cx_r, cy_r) = (palm_dets[[2, 0]], palm_dets[[3, 0]]); 
    // let size = palm_dets[[0, 0]];
    // let rotation = palm_dets[[1, 0]];
    let (size, rotation, cx_r, cy_r) = (palm_dets[[0, 0]], palm_rects[[4, 0]], palm_dets[[2, 0]], palm_dets[[3, 0]]); 
    // let mut ldmk_rels_global = calculate::rotate_points_around_z_axis(ldmk_rels_local.slice(s![0, .., ..]).clone().into_owned(), [0.5, 0.5], rotation); 
    let mut ldmk_rels_global = calculate::rotate_points_around_z_axis_and_scale(ldmk_rels_local.slice(s![0, .., ..]).clone().into_owned(), [0.5, 0.5], size / 1., rotation); 
    // println!("ldmk_rels_local (sliced): {:?}", ldmk_rels_local.slice(s![0, .., ..])); // 문제없음.
    ldmk_rels_global = calculate::translate_points(ldmk_rels_global, (cx_r - 0.5, cy_r - 0.5)); 


    let mut results: Vec<HandResult> = Vec::new();
    for i in 0..hand_count {
      let palm = PalmResult {
        cx: cx_r,
        cy: cy_r,
        size: size,
        rotation: rotation,
        // score: 
        handedness: "Unknown".to_string(),
        imshape: (original_width, original_height),
      };
      let ldmk = LandmarkResult {
        // coords: ldmk_rels_local.slice(s![i, .., ..]).clone().into_owned(),
        coords: ldmk_rels_global.clone().into_owned(),
      };
      results.push(HandResult { palm: palm, landmark: ldmk })
    }

    Ok(results)
    // Ok(())
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

use std::fs::File;
use std::io::BufReader;

#[derive(Deserialize, Debug)]
struct TestData {
  name: String,
  data: Vec<Vec<f32>>,
}

fn round_ndarray_elements(array: &ArrayD<f32>, decimal_places: i32) -> ArrayD<f32> {
  let factor = 10f32.powi(decimal_places);
  array.mapv(|elem| (elem * factor).round() / factor)
}

/// Test functionality
fn test() -> Result<(), Box<dyn Error>> {
  // User Params
  // Set test image
  // const TEST_NAME: &str = "hand";
  // const TEST_NAME: &str = "hand_90";
  // const TEST_NAME: &str = "hand_180";
  // const TEST_NAME: &str = "hand_270";
  // const TEST_NAME: &str = "bottom_left";
  const TEST_NAME: &str = "top_right";
  println!("TEST_NAME: {}", TEST_NAME);
  
  // image
  let test_img_path = format!("./tests/test_images/{}.jpg", TEST_NAME);
  let image = image::open(test_img_path)?;
  let palm_img = image.resize_exact(192, 192, CatmullRom);
  
  // model
  let hp = Handpose::new()?;
  
  // process
  let hand_results = hp.process(image.clone())?;
  println!("hand_results: {:?}", hand_results);
  
  // assess accuracy
  // read JSON file of ground-truth landmark points
  let file = File::open("./tests/hand_landmarks.json").expect("Failed to open test data file");
  let reader = BufReader::new(file);
  let data: Vec<TestData> = serde_json::from_reader(reader)?;
  let data_hashmap = data.iter().map(|x| (x.name.clone(), x.data.clone())).collect::<std::collections::HashMap<String, Vec<Vec<f32>>>>();
  // println!("data_hashmap: {:?}", data_hashmap);
  let ldmk_preds_ground_truth = data_hashmap.get(TEST_NAME).unwrap();
  // println!("ldmk_preds_ground_truth: {:?}", ldmk_preds_ground_truth);
  // calculate errors (abs & percentage)
  
  // perhaps I should flatten data before doing that
  let ldmk_preds_ground_truth_flat = ldmk_preds_ground_truth.iter().flatten().collect::<Vec<&f32>>();  
  
  // deref
  let ldmk_preds_ground_truth_flat = ldmk_preds_ground_truth_flat.iter().map(|x| **x).collect::<Vec<f32>>();

  // turn ldmk_preds_ground_truth into ndarrray
  let ldmk_preds_ground_truth_arr = Array::from_shape_vec((21, 3), ldmk_preds_ground_truth_flat).unwrap();
  println!("ldmk_preds_ground_truth_arr: \n {:?}", ldmk_preds_ground_truth_arr);
  
  let mut ldmk_coords_test = hand_results[0].landmark.coords.clone();
  // println!("ldmk_preds_test: \n {:?}", ldmk_preds_test);
  
  let ldmk_preds_test_rounded = round_ndarray_elements(&ldmk_coords_test.clone().into_dyn(), 4);
  println!("ldmk_preds_test_rounded: \n {:?}", ldmk_preds_test_rounded);
  
  let diff = ldmk_preds_ground_truth_arr.clone() - ldmk_preds_test_rounded.clone();
  let diff = diff.mapv(|elem| elem.abs()); // absolute value
  println!("diff: \n {:?}", round_ndarray_elements(&diff.clone().into_dyn(), 4));

  let diff_percentage = diff.clone() / ldmk_preds_ground_truth_arr.clone().mapv(|elem| elem.abs()) * 100.;
  println!("diff_percentage: {:?}", round_ndarray_elements(&diff_percentage.into_dyn(), 1));
  
  // visualize
  let scale_factor = 1.;
  // visualize::visualize2(image, palm_rects.slice(s![.., 0]).into_owned(), ldmk_preds_rel, scale_factor); // ALT1
  for hand in hand_results.iter() {
    let (cx_r, cy_r, size, rotation) = (hand.palm.cx, hand.palm.cy, hand.palm.size, hand.palm.rotation);
    // let pixmap = visualize::visualize2(palm_img.clone(), ldmk_coords_test.clone()); // ALT1
    let pixmap = visualize::visualize2(image.clone(), ldmk_coords_test.clone()); // ALT1
    pixmap.save_png("./viz/hand_viz.png");
  }
    

  Ok(())
}


fn main() -> Result<(), Box<dyn Error>>{
  #[cfg(target_arch = "wasm32")]
	ort::wasm::initialize();

  test()?;
  Ok(())
}

// NOTE let's develop post-processing / image manipulation step
// not by using ONNX but just with test data and tracing & modeling (understanding) of Python script
