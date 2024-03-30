#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]
#![allow(unused_mut)]

// basics
use std::error::Error;
use std::iter::Filter;
use std::path::{Path, PathBuf};
// arrays/vectors/tensors
use ndarray::{array, s, Array, Array1, Array2, Array3, Array4, Axis, ArrayBase};
use ndarray::{OwnedRepr, Dim, IxDyn};
use ort::{GraphOptimizationLevel, Session};
// images
use image::io::Reader as ImageReader;
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb, Rgba};
use image::imageops::FilterType;
use imageproc::drawing::draw_filled_rect_mut;
use imageproc::rect::Rect;
// use show_image::{event, AsImageView, WindowOptions};
mod calculate;


// #[derive(Debug, Clone, Copy)]
// struct BoundingBox {
//   x1: f32,
//   y1: f32,
//   x2: f32,
//   y2: f32
// }


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
  pub fn new(self) -> Result<Self, Box<dyn Error>>{
    let palm_file = PathBuf::from("./hand-gesture-recognition-using-onnx-main/model/palm_detection/palm_detection_full_inf_post_192x192.onnx");
    let palm_model = Session::builder()?
      .with_optimization_level(GraphOptimizationLevel::Level3)?
      .with_intra_threads(4)?
      // .with_model_from_file("./hand-gesture-recognition-using-onnx-main/model/palm_detection/palm_detection_full_inf_post_192x192.onnx")?;
      .with_model_from_file(&palm_file)?;
    let palm_size = 192;

    let ldmk_file = PathBuf::from("./hand-gesture-recognition-using-onnx-main/model/hand_landmark/hand_landmark_sparse_Nx3x224x224.onnx");
    let ldmk_model = Session::builder()?
      .with_optimization_level(GraphOptimizationLevel::Level3)?
      .with_intra_threads(4)?
      .with_model_from_file(&ldmk_file)?;
    let ldmk_size = 224;

    let detection_threshold = 0.5;

  //  Ok(())
  Ok( Self { palm_file, palm_model, palm_size, ldmk_file, ldmk_model, ldmk_size, detection_threshold })

  }

  pub fn process(self, image: DynamicImage) -> Result<(), Box<dyn Error>> {
    // TODO record width and height scales
    let (width_scale, height_scale) = image.dimensions();
    let palm_img = image.resize_exact(self.palm_size, self.palm_size, FilterType::CatmullRom);
    let palm_inputs = ort::inputs!["input" => self.image_to_onnx_input(image.clone()).view()]?; // NOTE this assumes array representation of image is no longer needed. if otherwise, separate into `palm_img_arr`` variable
    let palm_outputs = self.palm_model.run(palm_inputs)?;
    let palm_preds = palm_outputs["pdscore_boxx_boxy_boxsize_kp0x_kp0y_kp2x_kp2y"].extract_tensor::<f32>()?;
    let palm_preds_view = palm_preds.view().t().slice(s![.., ..]).into_owned();  
    let imshape: Array1<i32> = array![self.palm_size as i32, self.palm_size as i32]; // array![192, 192];
    let palm_dets = calculate::postprocess_palms(imshape.clone(), palm_preds_view, 0.5);
    // print!("{:?}", palm_dets);
    let rects = calculate::calculate_rects(palm_dets, imshape.clone());
    // print!("{:?}", rects);
    let rot_crops = calculate::rotate_and_crop_rectangle(&image, rects.clone());
    // print!("{:?}", rot_crops);
    let hand_count = rot_crops.len(); // TODO find a better implementation ?
    let mut ldmk_imgs: Vec<DynamicImage> = Vec::new();
    for i in 0..hand_count {
      let ldmk_img = rot_crops[i].resize_exact(self.ldmk_size, self.ldmk_size, FilterType::CatmullRom);
      ldmk_imgs.push(ldmk_img)
    }
    let ldmk_inputs = ort::inputs!["input" => self.images_to_onnx_input(ldmk_imgs).view()]?;
    let ldmk_outputs = self.ldmk_model.run(ldmk_inputs)?;
    let ldmk_preds = ldmk_outputs["xyz_x21"].extract_tensor::<f32>()?;
    let ldmk_preds_view = ldmk_preds.view().clone().into_owned();

    // let ldmk_preds_rel;
    // let ldmk_preds_abs;
    Ok(())
  }

  // fn image_to_onnx_input(self, image: DynamicImage) -> Array4<f32> {
  //   let mut img_arr = image.to_rgb8().into_vec();
  //   let (width, height) = image.dimensions();
  //   let channels = 3;
  //   // let mut onnx_input = Array::zeros((1, channels, height as i32, width as i32));
  //   let mut onnx_input = Array::zeros((1 as usize, channels as usize, height as usize, width as usize));
  //   for y in 0..height as usize {
  //     for x in 0..width as usize{
  //       let pixel = img_arr[y * width as usize + x];
  //       onnx_input[[0,0,y as usize,x as usize]] = pixel[0] as f32 / 255.0;
  //       onnx_input[[0,1,y as i32,x as i32]] = pixel[1] as f32 / 255.0;
  //       onnx_input[[0,2,y as i32,x as i32]] = pixel[2] as f32 / 255.0;
  //     }
  //   }
  //   onnx_input
  // }

  fn image_to_onnx_input(&self, image: DynamicImage) -> Array4<f32> {
    let mut img_arr = image.to_rgb8().into_vec();
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
    //   x_d = np.array(img_d).transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)/256 // HWC -> NCHW
  }

  // // refactor based on what's more readable!
  // let mut ldmk_image_array = Array4::<f32>::zeros((1, 3, LDMK_SIZE as _, LDMK_SIZE as _));
  // for (x, y, pixel) in ldmk_image.clone().into_rgb8().enumerate_pixels() {
  //   let (r, g, b) = (pixel[0], pixel[1], pixel[2]);
  //   // Set the RGB values in the array
  //   ldmk_image_array[[0, 0, y as _, x as _]] = (r as f32) / 255.;
  //   ldmk_image_array[[0, 1, y as _, x as _]] = (g as f32) / 255.;
  //   ldmk_image_array[[0, 2, y as _, x as _]] = (b as f32) / 255.;
  // }

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



// const TEST_IMAGE: &str = "hand.jpg";
// const TEST_IMAGE: &str = "bottom_left.jpg";
// const TEST_IMAGE: &str = "top_right.jpg";

// const TEST_NAME: &str = "hand";
// const TEST_NAME: &str = "hand_90";
const TEST_NAME: &str = "hand_180";
// const TEST_NAME: &str = "hand_270";
// const TEST_NAME: &str = "bottom_left";
// const TEST_NAME: &str = "top_right";

const PALM_SIZE: u32 = 192;
const LDMK_SIZE: u32 = 224;

fn read_image() -> Result<Array4<f32>, Box<dyn Error>> {
  // let img_path = Path::new("./tests/test_images/hand.jpg");
  // let img_path = Path::new(&format!("./tests/test_images/{}.jpg", TEST_NAME));
  let img_path = PathBuf::from(format!("./tests/test_images/{}.jpg", TEST_NAME));
  // let image = ImageReader::open(img_path)?.decode()?;
  let original_image = image::open(img_path)?;

  let image = original_image.resize_exact(PALM_SIZE, PALM_SIZE, FilterType::CatmullRom);

  let mut input = Array4::<f32>::zeros((1, 3, PALM_SIZE as _, PALM_SIZE as _));
  // Iterate over each pixel to fill the array
  for (x, y, pixel) in image.into_rgb8().enumerate_pixels() {
    let (r, g, b) = (pixel[0], pixel[1], pixel[2]);
    // Set the RGB values in the array
    input[[0, 0, y as _, x as _]] = (r as f32) / 255.;
    input[[0, 1, y as _, x as _]] = (g as f32) / 255.;
    input[[0, 2, y as _, x as _]] = (b as f32) / 255.;
  };

  return Ok(input);
}


// fn run() -> Result<Array, std::Box::dyn::error::Error>{
fn run() -> Result<(), Box<dyn Error>> {
  let palm_model = Session::builder()?
      .with_optimization_level(GraphOptimizationLevel::Level3)?
      .with_intra_threads(4)?
      .with_model_from_file("./hand-gesture-recognition-using-onnx-main/model/palm_detection/palm_detection_full_inf_post_192x192.onnx")?;
  
  let input = read_image().unwrap(); 
  // NOTE redundant
  // let img_path = Path::new("./tests/test_images/hand.jpg");
  let img_path = PathBuf::from(format!("./tests/test_images/{}.jpg", TEST_NAME));
  let original_image = image::open(img_path)?;
  let image = original_image.resize_exact(PALM_SIZE, PALM_SIZE, image::imageops::FilterType::CatmullRom);

  // let mut scale_factor = original_image.width() as f32 / PALM_SIZE as f32; // WARD

  let inputs = ort::inputs!["input" => input.view()]?;
  let outputs = palm_model.run(inputs)?;
  let preds = outputs["pdscore_boxx_boxy_boxsize_kp0x_kp0y_kp2x_kp2y"].extract_tensor::<f32>()?;
  
  // let output_view = output.view();
  let pred_view = preds.view().t().slice(s![.., 0]).into_owned();
  let preds_view = preds.view().t().slice(s![.., ..]).into_owned();

  for element in pred_view.iter() {
    // println!("{}", element);
  }

  let imshape: Array1<i32> = array![192, 192];

  let palm_dets = calculate::postprocess_palms(imshape.clone(), preds_view, 0.5);
  // print!("{:?}", palm_dets);

  let rects = calculate::calculate_rects(palm_dets, imshape.clone());
  // print!("{:?}", rects);

  let rot_crops = calculate::rotate_and_crop_rectangle(&image, rects.clone());
  // print!("{:?}", rot_crops);
  rot_crops[0].save("rot_crop.jpg")?;

  let ldmk_model = Session::builder()?
      .with_optimization_level(GraphOptimizationLevel::Level3)?
      .with_intra_threads(4)?
      .with_model_from_file("./hand-gesture-recognition-using-onnx-main/model/hand_landmark/hand_landmark_sparse_Nx3x224x224.onnx")?;

  let ldmk_image = rot_crops[0].resize_exact(LDMK_SIZE, LDMK_SIZE, image::imageops::FilterType::CatmullRom);
  // let ldmk_image = original_image.resize_exact(LDMK_SIZE, LDMK_SIZE, image::imageops::FilterType::CatmullRom);
  
  let mut ldmk_image_array = Array4::<f32>::zeros((1, 3, LDMK_SIZE as _, LDMK_SIZE as _));
  for (x, y, pixel) in ldmk_image.clone().into_rgb8().enumerate_pixels() {
    let (r, g, b) = (pixel[0], pixel[1], pixel[2]);
    // Set the RGB values in the array
    ldmk_image_array[[0, 0, y as _, x as _]] = (r as f32) / 255.;
    ldmk_image_array[[0, 1, y as _, x as _]] = (g as f32) / 255.;
    ldmk_image_array[[0, 2, y as _, x as _]] = (b as f32) / 255.;
  };
  let ldmk_inputs = ort::inputs!["input" => ldmk_image_array.view()]?;
  let ldmk_outputs = ldmk_model.run(ldmk_inputs)?;
  let ldmk_preds = ldmk_outputs["xyz_x21"].extract_tensor::<f32>()?;
  // println!("{:?}", ldmk_preds); // -> Tensor { data { array_view, strides, layout, dynamicity, ndim } }
  // let ldmk_pred_view = preds.view().t().slice(s![.., 0]).into_owned();
  // let ldmk_preds_view = preds.view().slice(s![.., ..]).into_owned(); // WRONG
  // let ldmk_preds_view: ArrayBase<OwnedRepr<f32>, Dim<[usize; 63]>> = preds.view().t().slice(s![.., ..]).into_owned();
  // println!("{:?}", ldmk_preds_view);
  // println!("{:?}", ldmk_preds.view().clone()); // WORKS
  // let ldmk_preds_view = ldmk_preds.view().clone().into_owned();
  // println!("{:?}", ldmk_preds_view);
  
  // let ldmk_preds_view_rel = calculate::calculate_relative(ldmk_preds_view.clone()); // TODO
  let ldmk_preds_rel = ldmk_preds_view.clone() / 192.;
  println!("{:?}", ldmk_preds_rel);
  
  // let ldmk_preds_0 = ldmk_outputs["hand_score"].extract_tensor::<f32>()?;
  // // println!("{:?}", ldmk_preds_0);
  // let ldmk_preds_0_view = ldmk_preds_0.view().slice(s![..,..]).into_owned();
  // println!("{:?}", ldmk_preds_0_view);

  // let mut scale_factor = original_image.width() as f32 / PALM_SIZE as f32; // w/ original image
  let mut scale_factor = 1.; // w/ palm_image
  let det_size = rects.get((3, 0)).unwrap();
  println!("Detection Size: {:?}", det_size);
  scale_factor *= det_size / 192.;
  println!("Scale Factor: {:?}", scale_factor);
  
  // let ldmk_preds_2 = ldmk_outputs["lefthand_0_or_righthand_1"].extract_tensor::<f32>()?;
  // // println!("{:?}", ldmk_preds_2);
  // let ldmk_preds_2_view = ldmk_preds_2.view().slice(s![..,..]).into_owned();
  // println!("{:?}", ldmk_preds_2_view);

  // visualize_ldmk_preds(ldmk_image.clone(), ldmk_preds_view); // WORKS
  // save(visualize_ldmk_preds(ldmk_preds_view));

  // TODO multiply rects by scale factor
  // visualize(original_image, rects.slice(s![.., 0]).into_owned(), ldmk_preds_view, scale_factor);
  // visualize(image, rects.slice(s![.., 0]).into_owned(), ldmk_preds_view, scale_factor); // ORIG
  visualize(image, rects.slice(s![.., 0]).into_owned(), ldmk_preds_rel, scale_factor); // ALT1
  // visualize_224_192(ldmk_image.resize_exact(192, 192, FilterType::CatmullRom), rects.slice(s![.., 0]).into_owned(), ldmk_preds_view, scale_factor); // 224 -> 192 unit test for conversion
  
  // visualize_crop_wo_rot(original_image, rects.slice(s![.., 0]).into_owned(), ldmk_preds_view, scale_factor); // NOTE WRONG
  // visualize_crop_wo_rot(ldmk_image.resize_exact(192, 192, FilterType::CatmullRom), rects.slice(s![.., 0]).into_owned(), ldmk_preds_view, scale_factor);

  Ok(())
  
}

// fn image_to_array(image: DynamicImage) -> Array3<f32> {
//   let mut image_array = Array4::<f32>::zeros((1, 3, LDMK_SIZE as _, LDMK_SIZE as _));
//   for (x, y, pixel) in image.into_rgb8().enumerate_pixels() {
//     let (r, g, b) = (pixel[0], pixel[1], pixel[2]);
//     // Set the RGB values in the array
//     image_array[[0, y as _, x as _]] = (r as f32) / 255.;
//     image_array[[1, y as _, x as _]] = (g as f32) / 255.;
//     image_array[[2, y as _, x as _]] = (b as f32) / 255.;
//   };
//   return image_array;
// }

fn dynamic_image_to_ndarray(image: DynamicImage) -> Array3<u8> { // ChatGPT
  // Convert the DynamicImage to an RgbImage (or another format as needed)
  let rgb_image = image.to_rgb8();

  // Get image dimensions
  let (width, height) = rgb_image.dimensions();

  // Extract raw pixels (flattened)
  let raw_pixels = rgb_image.into_raw(); // This is a Vec<u8>

  // Convert Vec<u8> to Array3<u8>
  let array = Array3::from_shape_vec((height as usize, width as usize, 3), raw_pixels)
      .expect("Error converting Vec<u8> to Array3<u8>");

  array
}

const FINGERS: [&str; 5] = ["thumb", "index", "middle", "ring", "pinky"];
const FINGER_COLORS: [(u8, u8, u8); 5] = [(180, 229, 255), (128, 64, 128), (0, 204, 255), (48, 255, 48), (192, 101, 21)];

const RADIUS: f32 = 3.0;

// fn visualize_ldmk_preds(ldmk_image: Array2<u8>, ldmk_pred: Array2<f32>) { // -> image::ImageBuffer<Rgba<u8> {
// fn visualize_ldmk_preds(ldmk_image: Array2<u8>, ldmk_pred: Array1<f32>) { // -> image::ImageBuffer<Rgba<u8> {
fn visualize_ldmk_preds(ldmk_image: DynamicImage, ldmk_pred: Array<f32, IxDyn>) { // -> image::ImageBuffer<Rgba<u8> {
  // // image
  let ldmk_image_array = dynamic_image_to_ndarray(ldmk_image.clone());
  let height = ldmk_image_array.shape()[0];
  let width = ldmk_image_array.shape()[1];

  let ldmk_pred_xyz = ldmk_pred.into_shape((21, 3)).unwrap();
  println!("{:?}", ldmk_pred_xyz);

  // let mut pixmap = Pixmap::new(height as u32, width as u32).unwrap(); // BLANK

  // FROM IMAGE (1) - doesn't work (yet)
  // let image_bytes = DynamicImage::into_bytes(ldmk_image.clone()); 
  // let image_buffer = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_vec(width as u32, height as u32, image_bytes[..].to_vec()).unwrap();
  // let mut pixmap = Pixmap::from_vec(image_buffer, IntSize::from_wh(width, height).unwrap()) 

  // FROM IMAGE (2)
  let img_rgba = ldmk_image.clone().to_rgba8(); // Ensure image is in RGBA format
  let mut pixmap = Pixmap::from_vec( // NOTE compiler warns
    img_rgba.into_raw(), // This converts the DynamicImage into a Vec<u8> of raw pixels
    IntSize::from_wh(width as u32, height as u32).unwrap()
  ).expect("Failed to create Pixmap");

  let thumb_paint = Paint {
    shader: Shader::SolidColor(Color::from_rgba8(180, 229, 255, 255)), // Green color
    anti_alias: true,
    ..Default::default()
  };
  let index_paint = Paint {
    shader: Shader::SolidColor(Color::from_rgba8(128, 64, 128, 255)), // Green color
    anti_alias: true,
    ..Default::default()
  };
  let middle_paint = Paint {
    shader: Shader::SolidColor(Color::from_rgba8(0, 204, 255, 255)), // Green color
    anti_alias: true,
    ..Default::default()
  };
  let ring_paint = Paint {
    shader: Shader::SolidColor(Color::from_rgba8(48, 255, 48, 255)), // Green color
    anti_alias: true,
    ..Default::default()
  };
  let pinky_paint = Paint {
    shader: Shader::SolidColor(Color::from_rgba8(192, 101, 21, 255)), // Green color
    anti_alias: true,
    ..Default::default()
  };

  let mut thumb_path_builder = PathBuilder::new();
  let mut index_path_builder = PathBuilder::new();
  let mut middle_path_builder = PathBuilder::new();
  let mut ring_path_builder = PathBuilder::new();
  let mut pinky_path_builder = PathBuilder::new();

  for (i, xyz) in ldmk_pred_xyz.axis_iter(Axis(0)).enumerate() {
    let x = xyz[0];
    let y = xyz[1];
    if i == 0 {
      thumb_path_builder.push_circle(x, y, RADIUS)
    }
    else if (i-1) / 4 == 0 {
      thumb_path_builder.push_circle(x, y, RADIUS)
    }
    else if (i-1) / 4 == 1 {
      index_path_builder.push_circle(x, y, RADIUS)
    }
    else if (i-1) / 4 == 2 {
      middle_path_builder.push_circle(x, y, RADIUS)
    }
    else if (i-1) / 4 == 3 {
      ring_path_builder.push_circle(x, y, RADIUS)
    }
    else {
      pinky_path_builder.push_circle(x, y, RADIUS)
    }
  }

  let thumb_path = thumb_path_builder.finish().unwrap();
  let index_path = index_path_builder.finish().unwrap();
  let middle_path = middle_path_builder.finish().unwrap();
  let ring_path = ring_path_builder.finish().unwrap();
  let pinky_path = pinky_path_builder.finish().unwrap();

  // Fill the path (circle) with the defined paint
  pixmap.fill_path(&thumb_path, &thumb_paint, FillRule::Winding, Transform::identity(), None);
  pixmap.fill_path(&index_path, &index_paint, FillRule::Winding, Transform::identity(), None);
  pixmap.fill_path(&middle_path, &middle_paint, FillRule::Winding, Transform::identity(), None);
  pixmap.fill_path(&ring_path, &ring_paint, FillRule::Winding, Transform::identity(), None);
  pixmap.fill_path(&pinky_path, &pinky_paint, FillRule::Winding, Transform::identity(), None);

  pixmap.save_png("hand_viz.png").unwrap();
}

use tiny_skia::*;

fn test_tiny_skia() {
  let mut pixmap = Pixmap::new(200, 200).unwrap();
  
  // Define paint for filling the circle
  let paint: Paint<'_> = Paint {
      shader: Shader::SolidColor(Color::from_rgba8(0, 128, 0, 255)), // Green color
      anti_alias: true,
      ..Default::default()
  };

  // Create a path for the circle
  let mut path_builder = PathBuilder::new();
  path_builder.push_circle(100.0, 100.0, 50.0); // Circle with center at (100, 100) and radius 50
  let path = path_builder.finish().unwrap();

  // Fill the path (circle) with the defined paint
  pixmap.fill_path(&path, &paint, FillRule::Winding, Transform::identity(), None);

  // Save the result to a file
  pixmap.save_png("filled_circle.png").unwrap();
}


// fn visualize(image: DynamicImage, ldmk_preds: Array2<f32>) {
// fn visualize(image: DynamicImage, palm_rect: Array<f32, IxDyn>, ldmk_pred: Array<f32, IxDyn>) {
// fn visualize(image: DynamicImage, palm_rect: Array1<f32>, ldmk_pred: Array<f32, IxDyn>, scale_factor: f32) { // ORIG
fn visualize(image: DynamicImage, palm_rect: Array1<f32>, ldmk_pred_rel: Array<f32, IxDyn>, scale_factor: f32) { // ALT1
  // /// IMAGE ///
  let image_array = dynamic_image_to_ndarray(image.clone());
  let height = image_array.shape()[0];
  let width = image_array.shape()[1];
  let img_rgba = image.clone().to_rgba8(); // Ensure image is in RGBA format
  // let mut pixmap = Pixmap::from_vec( // NOTE compiler warns
  let pixmap = Pixmap::from_vec(
    img_rgba.into_raw(), // This converts the DynamicImage into a Vec<u8> of raw pixels
    IntSize::from_wh(width as u32, height as u32).unwrap()
  ).expect("Failed to create Pixmap");
  
  // /// OUTPUT ///
  let img_rgba = image.clone().to_rgba8(); // Ensure image is in RGBA format
  let mut pixmap = Pixmap::from_vec(
    img_rgba.into_raw(), // This converts the DynamicImage into a Vec<u8> of raw pixels
    IntSize::from_wh(width as u32, height as u32).unwrap()
  ).expect("Failed to create Pixmap");

  // /// DATA ///
  // let mut ldmk_pred_xyz = ldmk_pred.into_shape((21, 3)).unwrap(); // ORIG
  let mut ldmk_pred_xyz = ldmk_pred_rel.into_shape((21, 3)).unwrap(); // ALT1
  // ldmk_pred_xyz *= 192. / 224.; // PALM_SIZE / LDMK_SIZE // ORIG
  // ldmk_pred_xyz /= 224.; // PALM_SIZE / LDMK_SIZE // ALT1

  let (cx, cy, a) = (palm_rect[0] * scale_factor, palm_rect[1] * scale_factor, palm_rect[4]);
  // let ldmk_pred_recovered = calculate::rotate_points_around_z_axis(ldmk_pred_xyz, [cx, cy], a);
  // let ldmk_pred_recovered = calculate::rotate_points_around_z_axis_and_scale(ldmk_pred_xyz, [cy, cx], scale_factor, a);
  // let mut ldmk_pred_recovered = calculate::rotate_points_around_z_axis_and_scale(ldmk_pred_xyz, [192./2., 192./2.], scale_factor, a); // ORIG
  // let mut ldmk_pred_recovered = calculate::rotate_points_around_z_axis_and_scale(ldmk_pred_xyz, [0.5, 0.5], scale_factor, a); // ALT1
  let mut ldmk_pred_recovered = calculate::rotate_points_around_z_axis(ldmk_pred_xyz, [palm_rect[0], palm_rect[1]], a); // ALT1
  // ldmk_pred_recovered = calculate::translate_points(ldmk_pred_recovered, (cx - 192./2., cy - 192./2.));
  // ldmk_pred_recovered = calculate::translate_points(ldmk_pred_recovered, (cx, cy));
  // ldmk_pred_recovered = calculate::translate_points(ldmk_pred_recovered, (0., 0.));
  // ldmk_pred_recovered = calculate::translate_points(ldmk_pred_recovered, (cy, cx));
  // ldmk_pred_recovered = calculate::translate_points(ldmk_pred_recovered, (-cy, -cx)); // 1
  // ldmk_pred_recovered = calculate::translate_points(ldmk_pred_recovered, (-cx, -cy)); // 2
  // ldmk_pred_recovered = calculate::translate_points(ldmk_pred_recovered, (-cx + 192./2., - cy + 192./2.)); // 3
  ldmk_pred_recovered = calculate::translate_points(ldmk_pred_recovered, (-cy + 192./2., - cx + 192./2.)); // 4

  // NOTE when moving, center-align!
  // experiment using CENTERPOINT & FRAMES ! -> into test cases!

  // /// PAINT /// 
  let thumb_paint = Paint {
    shader: Shader::SolidColor(Color::from_rgba8(180, 229, 255, 255)), // Green color
    anti_alias: true,
    ..Default::default()
  };
  let index_paint = Paint {
    shader: Shader::SolidColor(Color::from_rgba8(128, 64, 128, 255)), // Green color
    anti_alias: true,
    ..Default::default()
  };
  let middle_paint = Paint {
    shader: Shader::SolidColor(Color::from_rgba8(0, 204, 255, 255)), // Green color
    anti_alias: true,
    ..Default::default()
  };
  let ring_paint = Paint {
    shader: Shader::SolidColor(Color::from_rgba8(48, 255, 48, 255)), // Green color
    anti_alias: true,
    ..Default::default()
  };
  let pinky_paint = Paint {
    shader: Shader::SolidColor(Color::from_rgba8(192, 101, 21, 255)), // Green color
    anti_alias: true,
    ..Default::default()
  };

  let mut thumb_path_builder = PathBuilder::new();
  let mut index_path_builder = PathBuilder::new();
  let mut middle_path_builder = PathBuilder::new();
  let mut ring_path_builder = PathBuilder::new();
  let mut pinky_path_builder = PathBuilder::new();

  // TODO infer radius based on image size (make it dynamic!)

  for (i, xyz) in ldmk_pred_recovered.axis_iter(Axis(0)).enumerate() {
    let x = xyz[0];
    let y = xyz[1];
    if i == 0 {
      thumb_path_builder.push_circle(x, y, RADIUS)
    }
    else if (i-1) / 4 == 0 {
      thumb_path_builder.push_circle(x, y, RADIUS)
    }
    else if (i-1) / 4 == 1 {
      index_path_builder.push_circle(x, y, RADIUS)
    }
    else if (i-1) / 4 == 2 {
      middle_path_builder.push_circle(x, y, RADIUS)
    }
    else if (i-1) / 4 == 3 {
      ring_path_builder.push_circle(x, y, RADIUS)
    }
    else {
      pinky_path_builder.push_circle(x, y, RADIUS)
    }
  }

  let thumb_path = thumb_path_builder.finish().unwrap();
  let index_path = index_path_builder.finish().unwrap();
  let middle_path = middle_path_builder.finish().unwrap();
  let ring_path = ring_path_builder.finish().unwrap();
  let pinky_path = pinky_path_builder.finish().unwrap();

  // Fill the path (circle) with the defined paint
  pixmap.fill_path(&thumb_path, &thumb_paint, FillRule::Winding, Transform::identity(), None);
  pixmap.fill_path(&index_path, &index_paint, FillRule::Winding, Transform::identity(), None);
  pixmap.fill_path(&middle_path, &middle_paint, FillRule::Winding, Transform::identity(), None);
  pixmap.fill_path(&ring_path, &ring_paint, FillRule::Winding, Transform::identity(), None);
  pixmap.fill_path(&pinky_path, &pinky_paint, FillRule::Winding, Transform::identity(), None);

  pixmap.save_png("hand_viz.png").unwrap();

}
  
fn visualize_224_192(image: DynamicImage, palm_rect: Array1<f32>, ldmk_pred: Array<f32, IxDyn>, scale_factor: f32) {
  // /// IMAGE ///
  let image_array = dynamic_image_to_ndarray(image.clone());
  let height = image_array.shape()[0];
  let width = image_array.shape()[1];
  let img_rgba = image.clone().to_rgba8(); // Ensure image is in RGBA format
  // let mut pixmap = Pixmap::from_vec( // NOTE compiler warns
  let pixmap = Pixmap::from_vec(
    img_rgba.into_raw(), // This converts the DynamicImage into a Vec<u8> of raw pixels
    IntSize::from_wh(width as u32, height as u32).unwrap()
  ).expect("Failed to create Pixmap");
  
  // /// OUTPUT ///
  let img_rgba = image.clone().to_rgba8(); // Ensure image is in RGBA format
  let mut pixmap = Pixmap::from_vec(
    img_rgba.into_raw(), // This converts the DynamicImage into a Vec<u8> of raw pixels
    IntSize::from_wh(width as u32, height as u32).unwrap()
  ).expect("Failed to create Pixmap");

  // /// DATA ///
  let mut ldmk_pred_xyz = ldmk_pred.into_shape((21, 3)).unwrap();
  ldmk_pred_xyz *= 192. / 224.; // PALM_SIZE / LDMK_SIZE

  let (cx, cy, a) = (palm_rect[0] * scale_factor, palm_rect[1] * scale_factor, palm_rect[4]);
  // let ldmk_pred_recovered = calculate::rotate_points_around_z_axis(ldmk_pred_xyz, [cx, cy], a);
  let ldmk_pred_recovered = calculate::rotate_points_around_z_axis_and_scale(ldmk_pred_xyz, [cy, cx], scale_factor, 0.);

  // /// PAINT /// 
  let thumb_paint = Paint {
    shader: Shader::SolidColor(Color::from_rgba8(180, 229, 255, 255)), // Green color
    anti_alias: true,
    ..Default::default()
  };
  let index_paint = Paint {
    shader: Shader::SolidColor(Color::from_rgba8(128, 64, 128, 255)), // Green color
    anti_alias: true,
    ..Default::default()
  };
  let middle_paint = Paint {
    shader: Shader::SolidColor(Color::from_rgba8(0, 204, 255, 255)), // Green color
    anti_alias: true,
    ..Default::default()
  };
  let ring_paint = Paint {
    shader: Shader::SolidColor(Color::from_rgba8(48, 255, 48, 255)), // Green color
    anti_alias: true,
    ..Default::default()
  };
  let pinky_paint = Paint {
    shader: Shader::SolidColor(Color::from_rgba8(192, 101, 21, 255)), // Green color
    anti_alias: true,
    ..Default::default()
  };

  let mut thumb_path_builder = PathBuilder::new();
  let mut index_path_builder = PathBuilder::new();
  let mut middle_path_builder = PathBuilder::new();
  let mut ring_path_builder = PathBuilder::new();
  let mut pinky_path_builder = PathBuilder::new();

  // TODO infer radius based on image size (make it dynamic!)

  for (i, xyz) in ldmk_pred_recovered.axis_iter(Axis(0)).enumerate() {
    let x = xyz[0];
    let y = xyz[1];
    if i == 0 {
      thumb_path_builder.push_circle(x, y, RADIUS)
    }
    else if (i-1) / 4 == 0 {
      thumb_path_builder.push_circle(x, y, RADIUS)
    }
    else if (i-1) / 4 == 1 {
      index_path_builder.push_circle(x, y, RADIUS)
    }
    else if (i-1) / 4 == 2 {
      middle_path_builder.push_circle(x, y, RADIUS)
    }
    else if (i-1) / 4 == 3 {
      ring_path_builder.push_circle(x, y, RADIUS)
    }
    else {
      pinky_path_builder.push_circle(x, y, RADIUS)
    }
  }

  let thumb_path = thumb_path_builder.finish().unwrap();
  let index_path = index_path_builder.finish().unwrap();
  let middle_path = middle_path_builder.finish().unwrap();
  let ring_path = ring_path_builder.finish().unwrap();
  let pinky_path = pinky_path_builder.finish().unwrap();

  // Fill the path (circle) with the defined paint
  pixmap.fill_path(&thumb_path, &thumb_paint, FillRule::Winding, Transform::identity(), None);
  pixmap.fill_path(&index_path, &index_paint, FillRule::Winding, Transform::identity(), None);
  pixmap.fill_path(&middle_path, &middle_paint, FillRule::Winding, Transform::identity(), None);
  pixmap.fill_path(&ring_path, &ring_paint, FillRule::Winding, Transform::identity(), None);
  pixmap.fill_path(&pinky_path, &pinky_paint, FillRule::Winding, Transform::identity(), None);

  pixmap.save_png("hand_viz.png").unwrap();

}
  
fn visualize_crop_wo_rot(image: DynamicImage, palm_rect: Array1<f32>, ldmk_pred: Array<f32, IxDyn>, scale_factor: f32) {
  // /// IMAGE ///
  let image_array = dynamic_image_to_ndarray(image.clone());
  let height = image_array.shape()[0];
  let width = image_array.shape()[1];
  let img_rgba = image.clone().to_rgba8(); // Ensure image is in RGBA format
  // let mut pixmap = Pixmap::from_vec( // NOTE compiler warns
  let pixmap = Pixmap::from_vec(
    img_rgba.into_raw(), // This converts the DynamicImage into a Vec<u8> of raw pixels
    IntSize::from_wh(width as u32, height as u32).unwrap()
  ).expect("Failed to create Pixmap");
  
  // /// OUTPUT ///
  let img_rgba = image.clone().to_rgba8(); // Ensure image is in RGBA format
  let mut pixmap = Pixmap::from_vec(
    img_rgba.into_raw(), // This converts the DynamicImage into a Vec<u8> of raw pixels
    IntSize::from_wh(width as u32, height as u32).unwrap()
  ).expect("Failed to create Pixmap");

  // /// DATA ///
  let mut ldmk_pred_xyz = ldmk_pred.into_shape((21, 3)).unwrap();
  ldmk_pred_xyz *= 192. / 224.; // PALM_SIZE / LDMK_SIZE

  let (cx, cy, a) = (palm_rect[0], palm_rect[1], palm_rect[4]);
  let mut ldmk_pred_recovered = calculate::rotate_points_around_z_axis_and_scale(ldmk_pred_xyz, [192./2., 192./2.], scale_factor, a);
  // let ldmk_pred_recovered = calculate::rotate_points_around_z_axis_and_scale(ldmk_pred_xyz, [cy, cx], scale_factor, 0.);

  ldmk_pred_recovered = calculate::translate_points(ldmk_pred_recovered, (cx, cy));

  // /// PAINT /// 
  let thumb_paint = Paint {
    shader: Shader::SolidColor(Color::from_rgba8(180, 229, 255, 255)), // Green color
    anti_alias: true,
    ..Default::default()
  };
  let index_paint = Paint {
    shader: Shader::SolidColor(Color::from_rgba8(128, 64, 128, 255)), // Green color
    anti_alias: true,
    ..Default::default()
  };
  let middle_paint = Paint {
    shader: Shader::SolidColor(Color::from_rgba8(0, 204, 255, 255)), // Green color
    anti_alias: true,
    ..Default::default()
  };
  let ring_paint = Paint {
    shader: Shader::SolidColor(Color::from_rgba8(48, 255, 48, 255)), // Green color
    anti_alias: true,
    ..Default::default()
  };
  let pinky_paint = Paint {
    shader: Shader::SolidColor(Color::from_rgba8(192, 101, 21, 255)), // Green color
    anti_alias: true,
    ..Default::default()
  };

  let mut thumb_path_builder = PathBuilder::new();
  let mut index_path_builder = PathBuilder::new();
  let mut middle_path_builder = PathBuilder::new();
  let mut ring_path_builder = PathBuilder::new();
  let mut pinky_path_builder = PathBuilder::new();

  // TODO infer radius based on image size (make it dynamic!)

  for (i, xyz) in ldmk_pred_recovered.axis_iter(Axis(0)).enumerate() {
    let x = xyz[0];
    let y = xyz[1];
    if i == 0 {
      thumb_path_builder.push_circle(x, y, RADIUS)
    }
    else if (i-1) / 4 == 0 {
      thumb_path_builder.push_circle(x, y, RADIUS)
    }
    else if (i-1) / 4 == 1 {
      index_path_builder.push_circle(x, y, RADIUS)
    }
    else if (i-1) / 4 == 2 {
      middle_path_builder.push_circle(x, y, RADIUS)
    }
    else if (i-1) / 4 == 3 {
      ring_path_builder.push_circle(x, y, RADIUS)
    }
    else {
      pinky_path_builder.push_circle(x, y, RADIUS)
    }
  }

  let thumb_path = thumb_path_builder.finish().unwrap();
  let index_path = index_path_builder.finish().unwrap();
  let middle_path = middle_path_builder.finish().unwrap();
  let ring_path = ring_path_builder.finish().unwrap();
  let pinky_path = pinky_path_builder.finish().unwrap();

  // Fill the path (circle) with the defined paint
  pixmap.fill_path(&thumb_path, &thumb_paint, FillRule::Winding, Transform::identity(), None);
  pixmap.fill_path(&index_path, &index_paint, FillRule::Winding, Transform::identity(), None);
  pixmap.fill_path(&middle_path, &middle_paint, FillRule::Winding, Transform::identity(), None);
  pixmap.fill_path(&ring_path, &ring_paint, FillRule::Winding, Transform::identity(), None);
  pixmap.fill_path(&pinky_path, &pinky_paint, FillRule::Winding, Transform::identity(), None);

  pixmap.save_png("hand_viz.png").unwrap();

}
  


fn main() {
  // let mut img = ImageBuffer::new(100, 100);
  // let rect = Rect::at(20, 20).of_size(60, 60);
  // draw_filled_rect_mut(&mut img, rect, Rgba([255u8, 0, 0, 255])); // Red color
  // img.save("output_with_rect.png").unwrap();
  // test_tiny_skia();

  run().unwrap();
}


  // // Postprocessing
  // let output = outputs["output0"]
  //     .extract_tensor::<f32>()
  //     .unwrap()
  //     .view()
  //     .t()
  //     .slice(s![.., .., 0])
  //     .into_owned();

// fn exp() {

// }


// NOTE let's develop post-processing / image manipulation step
// not by using ONNX but just with test data and tracing & modeling (understanding) of Python script