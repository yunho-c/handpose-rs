#![allow(unused_mut)]
#![allow(unused_braces)]

use ndarray::{s, Array, Array1, Array2, Array3, Array4, Axis, ArrayBase, Dim, OwnedRepr, arr2};
use std::f32::consts::{PI};
use std::cmp::{min, max};
use image::{Rgb, Rgba, DynamicImage, GenericImageView, ImageBuffer};
use imageproc::{rect::Rect, geometric_transformations::{rotate, Interpolation::Bicubic}};

const SCORE_THRESHOLD: f32 = 0.50;
const SQUARE_STANDARD_SIZE: f32 = 192.;
const SQUARE_PADDING_HALF_SIZE: f32 = 0.;
const OPERATION_WHEN_CROPPING_OUT_OF_RANGE: &str = "padding";

fn normalize_radians(angle: f32) -> f32 {
  return angle - 2.0 * PI * ((angle + PI) / (2.0 * PI)).floor();
}


// fn postprocess_palms(shape: Array1<i32>, boxes: Array2<f32>, threshold: f32) -> Array1<f32> {
pub fn postprocess_palms(imshape: Array1<i32>, boxes: Array2<f32>, threshold: f32) -> Array2<f32> {
  // TODO make threshold arg optional? but honestly not an urgence, it's just for conciseness

  let image_height = imshape[0] as f32;
  let image_width = imshape[1] as f32;

  // keep = boxes[.., 0] > threshold;
  // let confs = boxes[.., 0];
  let confs = boxes.slice(s![0, ..]);
  // let keep = boxes.mapv(|x| x > threshold); // NOTES mistake/typo: boxes instead of confs
  let keep = confs.mapv(|x| x > threshold);
  // let keepIndices = keep.indexed_iter().filter(|(_, &x)| x).map(|(i, _)| i).collect::<Vec<usize>>();
  let keep_indices: Vec<_> = keep.indexed_iter().filter_map(|(index, &item)| if item { Some(index) } else { None }).collect();
  let boxes = boxes.select(Axis(1), &keep_indices);

  let hand_count = keep_indices.len();
  // let mut hands: Array2<f32> = Array::zeros((hand_count, 8));
  // let mut hands_vec: Vec<f32> = vec![0.; 4*hand_count]; // flat
  let mut hands_vec: Vec<(f32, f32, f32, f32)> = Vec::new(); // = vec![0.; 4*hand_count];

  // for (pred) in boxes.axis_iter(Axis(1)) {
  for pred in boxes.axis_iter(Axis(1)) {
    let (pd_score, box_x, box_y, box_size, kp0_x, kp0_y, kp2_x, kp2_y) = (pred[0], pred[1], pred[2], pred[3], pred[4], pred[5], pred[6], pred[7]);
    if box_size > 0. {
      let ke02_x = kp2_x - kp0_x;
      let kp02_y = kp2_y - kp0_y;
      let sqn_rr_size = 2.9 * box_size;
      let rotation = (0.5 * PI) - (-kp02_y).atan2(ke02_x);
      let rotation2 = normalize_radians(rotation); // NOTE maybe refactor for better readability/understandability
      let sqn_rr_center_x = box_x + 0.5*box_size*rotation.sin();
      let mut sqn_rr_center_y = box_y - 0.5*box_size*rotation.cos();
      sqn_rr_center_y = (sqn_rr_center_y * SQUARE_STANDARD_SIZE - SQUARE_PADDING_HALF_SIZE) / image_height;
      hands_vec.push((sqn_rr_size, rotation, sqn_rr_center_x, sqn_rr_center_y));
    }
  }
  let hands_vec_flat: Vec<f32> = hands_vec.into_iter().flat_map(|(a, b, c, d)| vec![a, b, c ,d]).collect(); // Flatten the Vec<(T, T)> into a Vec<T>
  let hands = Array2::from_shape_vec((4, hand_count), hands_vec_flat).unwrap();

  return hands;
}


pub fn calculate_rects(palm_dets: Array2<f32>, imshape: Array1<i32>) -> Array2<f32> {
  let w = imshape[1] as f32;
  let h = imshape[0] as f32;
  let w_i = imshape[1] as i32;
  let h_i = imshape[0] as i32;
  let wh_ratio = 1.; // NOTE not sure if it's ever not 1 (test and see!)

  let mut rects_vec: Vec<(f32, f32, f32, f32, f32)> = Vec::new();
  let hand_count = palm_dets.shape()[1];

  // Loop through each detected palm.
  for palm in palm_dets.axis_iter(Axis(1)) {
    // Extract details of the palm.
    let sqn_rr_size = palm[0];
    let rotation = palm[1];
    let sqn_rr_center_x = palm[2];
    let sqn_rr_center_y = palm[3];

    // Convert relative coordinates to actual pixel values.
    let cx = (sqn_rr_center_x * w) as i32;
    let cy = (sqn_rr_center_y * h) as i32;
    let xmin = ((sqn_rr_center_x - (sqn_rr_size / 2.)) * w) as i32;
    let xmax = ((sqn_rr_center_x + (sqn_rr_size / 2.)) * w) as i32;
    let ymin = ((sqn_rr_center_y - (sqn_rr_size * wh_ratio / 2.)) * h) as i32;
    let ymax = ((sqn_rr_center_y + (sqn_rr_size * wh_ratio / 2.)) * h) as i32;

    // Ensure coordinates do not exceed image boundaries.
    let xmin = max(0, xmin);
    let xmax = min(w_i, xmax);
    let ymin = max(0, ymin);
    let ymax = min(w_i, ymax);

    // Calculate rotation degree.
    let degree = rotation * 180. / PI;
    rects_vec.push((cx as f32, cy as f32, (xmax-xmin) as f32, (ymax-ymin) as f32, degree));
  }

  // Convert the list of rectangles to an ndarray.
  let rects_vec_flat: Vec<f32> = rects_vec.into_iter().flat_map(|(a, b, c, d, e)| vec![a, b, c ,d, e]).collect(); // Flatten the Vec<(T, T)> into a Vec<T>
  let rects = Array2::from_shape_vec((5, hand_count), rects_vec_flat).unwrap();

  return rects;
}




// fn rotate_and_crop_rectangle(image: Array2<i8>, rects_tmp: Array2<f32>) -> Array2<f32> {
//   // let mut rects = copy.deepcopy(rects_tmp);
//   let mut rects = rects_tmp.clone();

//   let rotated_croped_images = [];
//   let height = image.shape()[0];
//   let width = image.shape()[1];

//   // Determine if rect is inside the entire image
//   if OPERATION_WHEN_CROPPING_OUT_OF_RANGE == "padding" {
//     // let size = (int(math.sqrt(f32::powf(width, 2) + height ** 2)) + 2) * 2;
//     let size = (f32::sqrt((width as f32).powf(2.) + (height as f32).powf(2.)) as i32) + 2 * 2; // TODO confirm
//     let image = pad_image(
//         image=image,
//         resize_width=size,
//         resize_height=size,
//     );
//     // Calculate adjustments
//     let x_adjust = (size - width).abs() / 2.0;
//     let y_adjust = (size - height).abs() / 2.0;

//     for mut row in rects.axis_iter_mut(Axis(1)) {
//       row[0] += x_adjust;
//       row[1] += y_adjust;
//     }
//   }

//   // NOTE not implemented. 
//   // else if operation_when_cropping_out_of_range == 'ignore' {
//   //   let inside_or_outsides = is_inside_rect(
//   //       rects=rects,
//   //       width_of_outer_rect=width,
//   //       height_of_outer_rect=height,
//   //   )
//   //   rects = rects[inside_or_outsides, ...]

//   }

//   let rect_bbx_upright = bounding_box_from_rotated_rect(
//       rects=rects,
//   );

//   let rect_bbx_upright_images = crop_rectangle(
//       image=image,
//       rects=rect_bbx_upright,
//   );

//   let rotated_rect_bbx_upright_images = image_rotation_without_crop(
//       images=rect_bbx_upright_images,
//       angles=rects[..., 4:5],
//   );

//   for rotated_rect_bbx_upright_image, rect in zip(rotated_rect_bbx_upright_images, rects):
//       crop_cx = rotated_rect_bbx_upright_image.shape[1]//2
//       crop_cy = rotated_rect_bbx_upright_image.shape[0]//2
//       rect_width = int(rect[2])
//       rect_height = int(rect[3])

//       rotated_croped_images.append(
//           rotated_rect_bbx_upright_image[
//               crop_cy-rect_height//2:crop_cy+(rect_height-rect_height//2),
//               crop_cx-rect_width//2:crop_cx+(rect_width-rect_width//2),
//           ]
//       )

//   return rotated_croped_images
// }

// fn rotate_and_crop_rectangle2(image: Array2<i8>, rects_tmp: Array2<f32>) -> Array2<f32> { // 
//     let height = image.shape()[0];
//     let width = image.shape()[1];


// }

pub fn rotate_and_crop_rectangle( // ChatGPT
    image: &DynamicImage,
    // rects: &[(f32, f32, f32, f32, f32)],
    rects: Array2<f32>,
) -> Vec<DynamicImage> {
    let mut crops = Vec::new();
    // let imshape = image.shape();
    let image_height = image.height(); // as f32;
    let image_width = image.width(); // as f32;

    // for &(cx, cy, w, h, a) in rects {
    for rect in rects.axis_iter(Axis(1)) {
      let cx = rect[0];
      let cy = rect[1];
      let w = rect[2];
      let h = rect[3];
      let a = rect[4];
      
      // Calculate crop coordinates
      let y1 = (cy - h / 2.0) as u32;
      let y2 = (cy + h / 2.0) as u32;
      let x1 = (cx - w / 2.0) as u32;
      let x2 = (cx + w / 2.0) as u32;

      // Rotate the image around the center (cx, cy)
      // Note: `rotate_with_center` requires angle in radians and the center as (x, y) in image coordinate space
      // let rotated = rotate_about_center(image, a.to_radians(), (cx as u32, cy as u32), image::Rgba([0, 0, 0, 0]));
      let image_bytes = DynamicImage::into_bytes(image.clone());
      let image_buffer = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_vec(image_width, image_height, image_bytes[..].to_vec()).unwrap();
      let rotated_buffer: image::ImageBuffer<_, Vec<_>> = rotate(&image_buffer, (cx, cy), -a.to_radians(), Bicubic, Rgb([255u8, 0, 0]));
      let rotated = DynamicImage::ImageRgb8(rotated_buffer);

      // Crop the image to the specified rectangle
      // Note: Cropping must be done carefully to stay within image bounds
      let crop = rotated.crop_imm(x1.min(rotated.width() - 1),
                                  y1.min(rotated.height() - 1),
                                  (x2 - x1).min(rotated.width() - x1),
                                  (y2 - y1).min(rotated.height() - y1));

      // crops.push(crop.to_image());
      crops.push(crop);
    }

    return crops;
}

// fn main() { // ChatGPT
//     // Example usage
//     let image = image::open("path/to/your/image.png").unwrap();
//     let rects = vec![(100.0, 100.0, 50.0, 50.0, 45.0)]; // Example rectangle: (cx, cy, w, h, angle in degrees)

//     let crops = rotate_and_crop_rectangle(&image, &rects);

//     // For simplicity, just save the first crop as an example
//     crops[0].save("cropped_image.png").unwrap();
// }

pub fn rotate_points_around_z_axis(points: Array2<f32>, center: [f32; 2], angle_degrees: f32) -> Array2<f32> {
  let angle_radians = angle_degrees * PI / 180.0;
  let cos_theta = angle_radians.cos();
  let sin_theta = angle_radians.sin();

  // Rotation matrix for Z-axis
  let rotation_matrix = arr2(&[
      [cos_theta, -sin_theta, 0.0],
      [sin_theta, cos_theta, 0.0],
      [0.0, 0.0, 1.0],
  ]);

  // Calculate the center point
  // let center = points.mean_axis(ndarray::Axis(0)).expect("Cannot compute the mean");
  let center = arr2(&[[center[0], center[1]]]);
  // println!("Center: {:?}", center); // GOOD

  // Translate points to origin, rotate, and translate back
  // points.map_axis(ndarray::Axis(1), |point| {
  //     let translated_point = point - &center;
  //     let rotated_point = rotation_matrix.dot(&translated_point);
  //     rotated_point + &center
  // })
  // // points.map_axis(ndarray::Axis(1), |point| {
  // let mapped = points.map_axis(ndarray::Axis(0), |point| {
  //   let translated_point = point.clone().into_owned() - center.t().clone().into_owned();
  //     println!("Translated point: {:?}", translated_point.clone().view());
  //     translated_point
  // });

  let centered = points - &center;

  let rotated = rotation_matrix.dot(&centered.t()).t().into_owned();

  let recovered = rotated + center;

  // let dummy_return = arr2(&[
  //     [0., 0., 0.],
  //     [0., 0., 0.],
  //     [0., 0., 0.],
  //   ]);

  return recovered;
}

pub fn rotate_points_around_z_axis_and_scale(points: Array2<f32>, center: [f32; 2], scale: f32, angle_degrees: f32) -> Array2<f32> {
  let angle_radians = angle_degrees * PI / 180.0;
  let cos_theta = angle_radians.cos();
  let sin_theta = angle_radians.sin();

  // Rotation matrix for Z-axis
  let rotation_matrix = arr2(&[
      [cos_theta, -sin_theta, 0.0],
      [sin_theta, cos_theta, 0.0],
      [0.0, 0.0, 1.0],
  ]);

  let rotation_matrix = rotation_matrix * scale;

  // Calculate the center point
  let center = points.mean_axis(ndarray::Axis(0)).expect("Cannot compute the mean");
  // println!("Center: {:?}", center); // GOOD

  let centered = points - &center;

  let rotated = rotation_matrix.dot(&centered.t()).t().into_owned();

  let recovered = rotated + center;

  return recovered;
}

// #[test]
// fn test_rotate() {
//   println!("{:?}", rotate_points_around_z_axis(arr2(&[[1., 0., 0.,], [3., 0., 0.,]]), [], 90.));
// }

pub fn translate_points (points: Array2<f32>, adjustments: (f32, f32)) -> Array2<f32> {
  let adjustments_array = arr2(&[[adjustments.0, adjustments.1, 0.]]);
  let translated = points + adjustments_array;
  return translated;
}

