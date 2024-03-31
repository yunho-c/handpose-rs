// basics
use ndarray::{array, s, Array, Array1, Array2, Array3, Array4, ArrayBase, Axis, Ix2};
use ndarray::{OwnedRepr, Dim, IxDyn};
// image
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb, Rgba};
// graphics
use tiny_skia::*;
// internal imports
use crate::calculate;

// colors
const FINGERS: [&str; 5] = ["thumb", "index", "middle", "ring", "pinky"];
const FINGER_COLORS: [(u8, u8, u8); 5] = [(180, 229, 255), (128, 64, 128), (0, 204, 255), (48, 255, 48), (192, 101, 21)];

// visualization params
const RADIUS: f32 = 3.0;


// TODO potentially incorporate this into image -> ORT input (this is probably faster & more concise)
fn dynamic_image_to_ndarray(image: DynamicImage) -> Array3<u8> { // ChatGPT
  // Convert the DynamicImage to an RgbImage
  let rgb_image = image.to_rgb8();
  
  // Get image dimensions
  let (width, height) = rgb_image.dimensions();

  // Extract raw pixels (flattened)
  let raw_pixels = rgb_image.into_raw(); // This is a Vec<u8>

  // Convert Vec<u8> to Array3<u8>
  let array = Array3::from_shape_vec((height as usize, width as usize, 3), raw_pixels)
      .expect("Error converting Vec<u8> to Array3<u8>");

  return array;
}

// pub fn visualize_ldmk_preds(ldmk_image: Array2<u8>, ldmk_pred: Array2<f32>) { // -> image::ImageBuffer<Rgba<u8> {
// pub fn visualize_ldmk_preds(ldmk_image: Array2<u8>, ldmk_pred: Array1<f32>) { // -> image::ImageBuffer<Rgba<u8> {
pub fn visualize_ldmk_preds(ldmk_image: DynamicImage, ldmk_pred: Array<f32, IxDyn>) { // -> image::ImageBuffer<Rgba<u8> {
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


// pub fn visualize(image: DynamicImage, ldmk_preds: Array2<f32>) {
// pub fn visualize(image: DynamicImage, palm_rect: Array<f32, IxDyn>, ldmk_pred: Array<f32, IxDyn>) {
// pub fn visualize(image: DynamicImage, palm_rect: Array1<f32>, ldmk_pred: Array<f32, IxDyn>, scale_factor: f32) { // ORIG
pub fn visualize(image: DynamicImage, palm_rect: Array1<f32>, ldmk_pred_rel: Array<f32, IxDyn>, scale_factor: f32) { // ALT1
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
  // println!("ldmk_pred_xyz: {:?}", ldmk_pred_xyz);
  // ldmk_pred_xyz *= 192. / 224.; // PALM_SIZE / LDMK_SIZE // ORIG
  // ldmk_pred_xyz /= 224.; // PALM_SIZE / LDMK_SIZE // ALT1

  let (cx_r, cy_r) = (palm_rect[0] / 192., palm_rect[1] / 192.);
  let (cx, cy, a) = (palm_rect[0] * scale_factor, palm_rect[1] * scale_factor, palm_rect[4]);
  // let ldmk_pred_recovered = calculate::rotate_points_around_z_axis(ldmk_pred_xyz, [cx, cy], a);
  // let ldmk_pred_recovered = calculate::rotate_points_around_z_axis_and_scale(ldmk_pred_xyz, [cy, cx], scale_factor, a);
  // let mut ldmk_pred_recovered = calculate::rotate_points_around_z_axis_and_scale(ldmk_pred_xyz, [192./2., 192./2.], scale_factor, a); // ORIG
  let mut ldmk_pred_recovered = calculate::rotate_points_around_z_axis_and_scale(ldmk_pred_xyz, [0.5, 0.5], scale_factor, a); // ALT1
  // println!("ldmk_pred_recovered: {:?}", ldmk_pred_recovered);
  // let mut ldmk_pred_recovered = calculate::rotate_points_around_z_axis(ldmk_pred_xyz, [palm_rect[0], palm_rect[1]], a); // ALT1
  // ldmk_pred_recovered = calculate::translate_points(ldmk_pred_recovered, (cx - 192./2., cy - 192./2.));
  // ldmk_pred_recovered = calculate::translate_points(ldmk_pred_recovered, (cx, cy));
  // ldmk_pred_recovered = calculate::translate_points(ldmk_pred_recovered, (0., 0.));
  // ldmk_pred_recovered = calculate::translate_points(ldmk_pred_recovered, (cy, cx));
  // ldmk_pred_recovered = calculate::translate_points(ldmk_pred_recovered, (-cy, -cx)); // 1
  // ldmk_pred_recovered = calculate::translate_points(ldmk_pred_recovered, (-cx, -cy)); // 2
  // ldmk_pred_recovered = calculate::translate_points(ldmk_pred_recovered, (-cx + 192./2., - cy + 192./2.)); // 3
  // ldmk_pred_recovered = calculate::translate_points(ldmk_pred_recovered, (-cy + 192./2., - cx + 192./2.)); // 4
  ldmk_pred_recovered = calculate::translate_points(ldmk_pred_recovered, (-cy_r + 0.5, - cx_r + 0.5)); // 4
  ldmk_pred_recovered = ldmk_pred_recovered * 192.;
  println!("ldmk_pred_recovered: {:?}", ldmk_pred_recovered);

  // NOTE when moving, center-align!
  // experiment using CENTERPOINT & FRAMES ! -> into test cases!

  // /// PAINT /// 
  let thumb_paint = Paint {
    shader: Shader::SolidColor(Color::from_rgba8(180, 229, 255, 255)), // Green color
    ..Default::default()
  };
  let index_paint = Paint {
    shader: Shader::SolidColor(Color::from_rgba8(128, 64, 128, 255)), // Green color
    ..Default::default()
  };
  let middle_paint = Paint {
    shader: Shader::SolidColor(Color::from_rgba8(0, 204, 255, 255)), // Green color
    ..Default::default()
  };
  let ring_paint = Paint {
    shader: Shader::SolidColor(Color::from_rgba8(48, 255, 48, 255)), // Green color
    ..Default::default()
  };
  let pinky_paint = Paint {
    shader: Shader::SolidColor(Color::from_rgba8(192, 101, 21, 255)), // Green color
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


pub fn draw(ldmk_pred_recovered: Array2<f32>, pixmap: &mut Pixmap) {
  let thumb_paint = Paint {
    shader: Shader::SolidColor(Color::from_rgba8(180, 229, 255, 255)), // Green color
    ..Default::default()
  };
  let index_paint = Paint {
    shader: Shader::SolidColor(Color::from_rgba8(128, 64, 128, 255)), // Green color
    ..Default::default()
  };
  let middle_paint = Paint {
    shader: Shader::SolidColor(Color::from_rgba8(0, 204, 255, 255)), // Green color
    ..Default::default()
  };
  let ring_paint = Paint {
    shader: Shader::SolidColor(Color::from_rgba8(48, 255, 48, 255)), // Green color
    ..Default::default()
  };
  let pinky_paint = Paint {
    shader: Shader::SolidColor(Color::from_rgba8(192, 101, 21, 255)), // Green color
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
}

// pub fn visualize2(image: DynamicImage, points_rel: Array<f32, Ix2>) { // ALT1
pub fn visualize2(image: DynamicImage, points_rel: Array<f32, Ix2>) -> Pixmap { // ALT1
  // /// IMAGE ///
  let image_array = dynamic_image_to_ndarray(image.clone());
  let height = image_array.shape()[0];
  let width = image_array.shape()[1];
  let img_rgba = image.clone().to_rgba8(); // Ensure image is in RGBA format
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
  let points_rel = points_rel * height as f32;

  // /// PAINT /// 
  draw(points_rel, &mut pixmap);

  // pixmap.save_png("hand_viz.png").unwrap();
  return pixmap;

}
  
// pub fn visualize_224_192(image: DynamicImage, palm_rect: Array1<f32>, ldmk_pred: Array<f32, IxDyn>, scale_factor: f32) {
//   // /// IMAGE ///
//   let image_array = dynamic_image_to_ndarray(image.clone());
//   let height = image_array.shape()[0];
//   let width = image_array.shape()[1];
//   let img_rgba = image.clone().to_rgba8(); // Ensure image is in RGBA format
//   // let mut pixmap = Pixmap::from_vec( // NOTE compiler warns
//   let pixmap = Pixmap::from_vec(
//     img_rgba.into_raw(), // This converts the DynamicImage into a Vec<u8> of raw pixels
//     IntSize::from_wh(width as u32, height as u32).unwrap()
//   ).expect("Failed to create Pixmap");
  
//   // /// OUTPUT ///
//   let img_rgba = image.clone().to_rgba8(); // Ensure image is in RGBA format
//   let mut pixmap = Pixmap::from_vec(
//     img_rgba.into_raw(), // This converts the DynamicImage into a Vec<u8> of raw pixels
//     IntSize::from_wh(width as u32, height as u32).unwrap()
//   ).expect("Failed to create Pixmap");

//   // /// DATA ///
//   let mut ldmk_pred_xyz = ldmk_pred.into_shape((21, 3)).unwrap();
//   ldmk_pred_xyz *= 192. / 224.; // PALM_SIZE / LDMK_SIZE

//   let (cx, cy, a) = (palm_rect[0] * scale_factor, palm_rect[1] * scale_factor, palm_rect[4]);
//   // let ldmk_pred_recovered = calculate::rotate_points_around_z_axis(ldmk_pred_xyz, [cx, cy], a);
//   let ldmk_pred_recovered = calculate::rotate_points_around_z_axis_and_scale(ldmk_pred_xyz, [cy, cx], scale_factor, 0.);

//   // /// PAINT /// 
//   let thumb_paint = Paint {
//     shader: Shader::SolidColor(Color::from_rgba8(180, 229, 255, 255)), // Green color
//     anti_alias: true,
//     ..Default::default()
//   };
//   let index_paint = Paint {
//     shader: Shader::SolidColor(Color::from_rgba8(128, 64, 128, 255)), // Green color
//     anti_alias: true,
//     ..Default::default()
//   };
//   let middle_paint = Paint {
//     shader: Shader::SolidColor(Color::from_rgba8(0, 204, 255, 255)), // Green color
//     anti_alias: true,
//     ..Default::default()
//   };
//   let ring_paint = Paint {
//     shader: Shader::SolidColor(Color::from_rgba8(48, 255, 48, 255)), // Green color
//     anti_alias: true,
//     ..Default::default()
//   };
//   let pinky_paint = Paint {
//     shader: Shader::SolidColor(Color::from_rgba8(192, 101, 21, 255)), // Green color
//     anti_alias: true,
//     ..Default::default()
//   };

//   let mut thumb_path_builder = PathBuilder::new();
//   let mut index_path_builder = PathBuilder::new();
//   let mut middle_path_builder = PathBuilder::new();
//   let mut ring_path_builder = PathBuilder::new();
//   let mut pinky_path_builder = PathBuilder::new();

//   // TODO infer radius based on image size (make it dynamic!)

//   for (i, xyz) in ldmk_pred_recovered.axis_iter(Axis(0)).enumerate() {
//     let x = xyz[0];
//     let y = xyz[1];
//     if i == 0 {
//       thumb_path_builder.push_circle(x, y, RADIUS)
//     }
//     else if (i-1) / 4 == 0 {
//       thumb_path_builder.push_circle(x, y, RADIUS)
//     }
//     else if (i-1) / 4 == 1 {
//       index_path_builder.push_circle(x, y, RADIUS)
//     }
//     else if (i-1) / 4 == 2 {
//       middle_path_builder.push_circle(x, y, RADIUS)
//     }
//     else if (i-1) / 4 == 3 {
//       ring_path_builder.push_circle(x, y, RADIUS)
//     }
//     else {
//       pinky_path_builder.push_circle(x, y, RADIUS)
//     }
//   }

//   let thumb_path = thumb_path_builder.finish().unwrap();
//   let index_path = index_path_builder.finish().unwrap();
//   let middle_path = middle_path_builder.finish().unwrap();
//   let ring_path = ring_path_builder.finish().unwrap();
//   let pinky_path = pinky_path_builder.finish().unwrap();

//   // Fill the path (circle) with the defined paint
//   pixmap.fill_path(&thumb_path, &thumb_paint, FillRule::Winding, Transform::identity(), None);
//   pixmap.fill_path(&index_path, &index_paint, FillRule::Winding, Transform::identity(), None);
//   pixmap.fill_path(&middle_path, &middle_paint, FillRule::Winding, Transform::identity(), None);
//   pixmap.fill_path(&ring_path, &ring_paint, FillRule::Winding, Transform::identity(), None);
//   pixmap.fill_path(&pinky_path, &pinky_paint, FillRule::Winding, Transform::identity(), None);

//   pixmap.save_png("hand_viz.png").unwrap();

// }
  
// pub fn visualize_crop_wo_rot(image: DynamicImage, palm_rect: Array1<f32>, ldmk_pred: Array<f32, IxDyn>, scale_factor: f32) {
//   // /// IMAGE ///
//   let image_array = dynamic_image_to_ndarray(image.clone());
//   let height = image_array.shape()[0];
//   let width = image_array.shape()[1];
//   let img_rgba = image.clone().to_rgba8(); // Ensure image is in RGBA format
//   // let mut pixmap = Pixmap::from_vec( // NOTE compiler warns
//   let pixmap = Pixmap::from_vec(
//     img_rgba.into_raw(), // This converts the DynamicImage into a Vec<u8> of raw pixels
//     IntSize::from_wh(width as u32, height as u32).unwrap()
//   ).expect("Failed to create Pixmap");
  
//   // /// OUTPUT ///
//   let img_rgba = image.clone().to_rgba8(); // Ensure image is in RGBA format
//   let mut pixmap = Pixmap::from_vec(
//     img_rgba.into_raw(), // This converts the DynamicImage into a Vec<u8> of raw pixels
//     IntSize::from_wh(width as u32, height as u32).unwrap()
//   ).expect("Failed to create Pixmap");

//   // /// DATA ///
//   let mut ldmk_pred_xyz = ldmk_pred.into_shape((21, 3)).unwrap();
//   ldmk_pred_xyz *= 192. / 224.; // PALM_SIZE / LDMK_SIZE

//   let (cx, cy, a) = (palm_rect[0], palm_rect[1], palm_rect[4]);
//   let mut ldmk_pred_recovered = calculate::rotate_points_around_z_axis_and_scale(ldmk_pred_xyz, [192./2., 192./2.], scale_factor, a);
//   // let ldmk_pred_recovered = calculate::rotate_points_around_z_axis_and_scale(ldmk_pred_xyz, [cy, cx], scale_factor, 0.);

//   ldmk_pred_recovered = calculate::translate_points(ldmk_pred_recovered, (cx, cy));

//   // /// PAINT /// 
//   let thumb_paint = Paint {
//     shader: Shader::SolidColor(Color::from_rgba8(180, 229, 255, 255)), // Green color
//     anti_alias: true,
//     ..Default::default()
//   };
//   let index_paint = Paint {
//     shader: Shader::SolidColor(Color::from_rgba8(128, 64, 128, 255)), // Green color
//     anti_alias: true,
//     ..Default::default()
//   };
//   let middle_paint = Paint {
//     shader: Shader::SolidColor(Color::from_rgba8(0, 204, 255, 255)), // Green color
//     anti_alias: true,
//     ..Default::default()
//   };
//   let ring_paint = Paint {
//     shader: Shader::SolidColor(Color::from_rgba8(48, 255, 48, 255)), // Green color
//     anti_alias: true,
//     ..Default::default()
//   };
//   let pinky_paint = Paint {
//     shader: Shader::SolidColor(Color::from_rgba8(192, 101, 21, 255)), // Green color
//     anti_alias: true,
//     ..Default::default()
//   };

//   let mut thumb_path_builder = PathBuilder::new();
//   let mut index_path_builder = PathBuilder::new();
//   let mut middle_path_builder = PathBuilder::new();
//   let mut ring_path_builder = PathBuilder::new();
//   let mut pinky_path_builder = PathBuilder::new();

//   // TODO infer radius based on image size (make it dynamic!)

//   for (i, xyz) in ldmk_pred_recovered.axis_iter(Axis(0)).enumerate() {
//     let x = xyz[0];
//     let y = xyz[1];
//     if i == 0 {
//       thumb_path_builder.push_circle(x, y, RADIUS)
//     }
//     else if (i-1) / 4 == 0 {
//       thumb_path_builder.push_circle(x, y, RADIUS)
//     }
//     else if (i-1) / 4 == 1 {
//       index_path_builder.push_circle(x, y, RADIUS)
//     }
//     else if (i-1) / 4 == 2 {
//       middle_path_builder.push_circle(x, y, RADIUS)
//     }
//     else if (i-1) / 4 == 3 {
//       ring_path_builder.push_circle(x, y, RADIUS)
//     }
//     else {
//       pinky_path_builder.push_circle(x, y, RADIUS)
//     }
//   }

//   let thumb_path = thumb_path_builder.finish().unwrap();
//   let index_path = index_path_builder.finish().unwrap();
//   let middle_path = middle_path_builder.finish().unwrap();
//   let ring_path = ring_path_builder.finish().unwrap();
//   let pinky_path = pinky_path_builder.finish().unwrap();

//   // Fill the path (circle) with the defined paint
//   pixmap.fill_path(&thumb_path, &thumb_paint, FillRule::Winding, Transform::identity(), None);
//   pixmap.fill_path(&index_path, &index_paint, FillRule::Winding, Transform::identity(), None);
//   pixmap.fill_path(&middle_path, &middle_paint, FillRule::Winding, Transform::identity(), None);
//   pixmap.fill_path(&ring_path, &ring_paint, FillRule::Winding, Transform::identity(), None);
//   pixmap.fill_path(&pinky_path, &pinky_paint, FillRule::Winding, Transform::identity(), None);

//   pixmap.save_png("hand_viz.png").unwrap();

// }

// /// =========
// /// Tests
// /// =========
// pub fn test_tiny_skia() {
//   let mut pixmap = Pixmap::new(200, 200).unwrap();
  
//   // Define paint for filling the circle
//   let paint: Paint<'_> = Paint {
//       shader: Shader::SolidColor(Color::from_rgba8(0, 128, 0, 255)), // Green color
//       anti_alias: true,
//       ..Default::default()
//   };

//   // Create a path for the circle
//   let mut path_builder = PathBuilder::new();
//   path_builder.push_circle(100.0, 100.0, 50.0); // Circle with center at (100, 100) and radius 50
//   let path = path_builder.finish().unwrap();

//   // Fill the path (circle) with the defined paint
//   pixmap.fill_path(&path, &paint, FillRule::Winding, Transform::identity(), None);

//   // Save the result to a file
//   pixmap.save_png("filled_circle.png").unwrap();
// }
