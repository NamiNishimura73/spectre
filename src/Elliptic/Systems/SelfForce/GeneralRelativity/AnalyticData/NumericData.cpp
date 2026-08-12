// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/SelfForce/GeneralRelativity/AnalyticData/NumericData.hpp"

#include <algorithm>
#include <complex>
#include <cstddef>
// #include <effsource_gr.hpp>
#include <effsource_comoving.hpp>
#include <utility>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/AnalyticData/CircularOrbitConvertEffsource.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/Tags.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/Helpers.hpp"
#include "NumericalAlgorithms/Interpolation/MultiLinearSpanInterpolation.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "PointwiseFunctions/GeneralRelativity/TortoiseCoordinates.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/CaptureForError.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

namespace GrSelfForce::AnalyticData {

namespace {

Interpolator load_data_from_file(const std::string& filename,
                                 const std::string& subfile_name) {
  // Open file
  CAPTURE_FOR_ERROR(filename);
  CAPTURE_FOR_ERROR(subfile_name);
  const h5::H5File<h5::AccessType::ReadOnly> h5file(filename);
  const auto& datfile = h5file.get<h5::Dat>(subfile_name);

  // Read metadata
  const auto num_radial_points =
      h5::read_value_attribute<size_t>(datfile.dataset_id(), "rNum");
  const auto num_angular_points =
      h5::read_value_attribute<size_t>(datfile.dataset_id(), "thetaNum");
  const auto delta_r =
      h5::read_value_attribute<double>(datfile.dataset_id(), "dr");
  const auto r_min =
      h5::read_value_attribute<double>(datfile.dataset_id(), "rMin");
  const auto delta_theta =
      h5::read_value_attribute<double>(datfile.dataset_id(), "dtheta");
  const auto theta_min =
      h5::read_value_attribute<double>(datfile.dataset_id(), "thetaMin");

  // Construct coordinates
  std::vector<double> r(num_radial_points);
  for (size_t i = 0; i < num_radial_points; ++i) {
    r[i] = r_min + static_cast<double>(i) * delta_r;
  }
  std::vector<double> theta(num_angular_points);
  for (size_t j = 0; j < num_angular_points; ++j) {
    theta[j] = theta_min + static_cast<double>(j) * delta_theta;
  }

  // Load data
  static constexpr size_t NumberOfVars = 20;
  const auto matrix_data = datfile.get_data();
  if (matrix_data.rows() != num_radial_points * num_angular_points) {
    const size_t rows = matrix_data.rows();
    CAPTURE_FOR_ERROR(rows);
    CAPTURE_FOR_ERROR(num_radial_points);
    CAPTURE_FOR_ERROR(num_angular_points);
    ERROR("Number of points in data file does not match header information");
  }
  if (matrix_data.columns() != NumberOfVars) {
    const size_t columns = matrix_data.columns();
    CAPTURE_FOR_ERROR(columns);
    CAPTURE_FOR_ERROR(NumberOfVars);
    ERROR("Unexpected number of variables in data file");
  }
  // Data is stored in (var, r, theta) order with var varying fastest
  std::vector<double> flat_data(NumberOfVars * num_radial_points *
                                num_angular_points);
  for (size_t i = 0; i < num_radial_points * num_angular_points; i++) {
    for (size_t k = 0; k < NumberOfVars; ++k) {
      flat_data[i * NumberOfVars + k] = matrix_data(i, k);
    }
  }

  // Close file
  h5file.close();

  // Construct interpolator
  return {std::move(r), std::move(theta), std::move(flat_data)};
}

Interpolator1D load_data_from_file_1D(const std::string& filename,
                                      const std::string& subfile_name,
                                      const std::string& num_points_attr,
                                      const std::string& coord_min_attr,
                                      const std::string& delta_coord_attr) {
  CAPTURE_FOR_ERROR(filename);
  CAPTURE_FOR_ERROR(subfile_name);
  const h5::H5File<h5::AccessType::ReadOnly> h5file(filename);
  const auto& datfile = h5file.get<h5::Dat>(subfile_name);

  const auto num_points =
      h5::read_value_attribute<size_t>(datfile.dataset_id(), num_points_attr);
  const auto coord_min =
      h5::read_value_attribute<double>(datfile.dataset_id(), coord_min_attr);
  const auto delta_coord =
      h5::read_value_attribute<double>(datfile.dataset_id(), delta_coord_attr);

  std::vector<double> coord(num_points);
  for (size_t i = 0; i < num_points; ++i) {
    coord[i] = coord_min + static_cast<double>(i) * delta_coord;
  }

  static constexpr size_t NumberOfVars = 40;
  const auto matrix_data = datfile.get_data();
  if (matrix_data.rows() != num_points) {
    const size_t rows = matrix_data.rows();
    CAPTURE_FOR_ERROR(rows);
    CAPTURE_FOR_ERROR(num_points);
    ERROR("Number of points in data file does not match header information");
  }
  if (matrix_data.columns() != NumberOfVars) {
    const size_t columns = matrix_data.columns();
    CAPTURE_FOR_ERROR(columns);
    CAPTURE_FOR_ERROR(NumberOfVars);
    ERROR("Unexpected number of variables in data file");
  }
  std::vector<double> flat_data(NumberOfVars * num_points);
  for (size_t i = 0; i < num_points; ++i) {
    for (size_t k = 0; k < NumberOfVars; ++k) {
      flat_data[i * NumberOfVars + k] = matrix_data(i, k);
    }
  }

  h5file.close();
  return {std::move(coord), std::move(flat_data)};
}

std::array<Interpolator, 7> load_all_data(const std::string& filename) {
  return {{load_data_from_file(filename, "RetRetV"),
           load_data_from_file(filename, "RetRetT"),
           load_data_from_file(filename, "RetRetU"),
           load_data_from_file(filename, "Seff"),
           load_data_from_file(filename, "Puncture"),
           load_data_from_file(filename, "drPuncture"),
           load_data_from_file(filename, "dthPuncture")}};
}

std::array<Interpolator1D, 4> load_all_boundary_data(
    const std::string& filename) {
  return {{load_data_from_file_1D(filename, "Left", "thetaNum", "thetaMin",
                                  "dtheta"),
           load_data_from_file_1D(filename, "Right", "thetaNum", "thetaMin",
                                  "dtheta"),
           load_data_from_file_1D(filename, "Bottom", "rNum", "rMin", "dr"),
           load_data_from_file_1D(filename, "Top", "rNum", "rMin", "dr")}};
}

}  // namespace

NumericData::NumericData(
    std::string filename, const double black_hole_mass,
    const double black_hole_spin, const double orbital_radius,
    const int m_mode_number,
    const std::array<double, 4> hyperboloidal_slicing_transitions,
    const bool penetrating_horizon, const int version, const bool pi_2_rotation)
    : filename_(std::move(filename)),
      circular_orbit_(black_hole_mass, black_hole_spin, orbital_radius,
                      m_mode_number,
                      {{{hyperboloidal_slicing_transitions[0],
                         hyperboloidal_slicing_transitions[1],
                         hyperboloidal_slicing_transitions[2],
                         hyperboloidal_slicing_transitions[3]}}},
                      penetrating_horizon, version),
      pi_2_rotation_(pi_2_rotation) {
  interpolators_ = load_all_data(filename_);
  boundary_interpolators_ = load_all_boundary_data(filename_);
}

NumericData::NumericData(CkMigrateMessage* m)
    : elliptic::analytic_data::Background(m),
      elliptic::analytic_data::InitialGuess(m) {}

tnsr::I<double, 2> NumericData::puncture_position() const {
  return circular_orbit_.puncture_position();
}

// Background
tuples::TaggedTuple<Tags::Alpha, Tags::Beta, Tags::GammaRstar, Tags::GammaTheta>
NumericData::variables(
    const tnsr::I<DataVector, 2>& x,
    tmpl::list<Tags::Alpha, Tags::Beta, Tags::GammaRstar, Tags::GammaTheta>
        meta) const {
  return circular_orbit_.variables(x, meta);
}

// Initial guess
tuples::TaggedTuple<Tags::MMode> NumericData::variables(
    const tnsr::I<DataVector, 2>& x, tmpl::list<Tags::MMode> meta) const {
  return circular_orbit_.variables(x, meta);
}

// Fixed sources
tuples::TaggedTuple<
    ::Tags::FixedSource<Tags::MMode>, Tags::SingularField,
    ::Tags::deriv<Tags::SingularField, tmpl::size_t<2>, Frame::Inertial>,
    Tags::BoyerLindquistRadius, Tags::RawEffSource, Tags::EF_EffSource,
    Tags::RawPuncture, Tags::EF_Puncture>
NumericData::variables(
    const tnsr::I<DataVector, 2>& x,
    tmpl::list<
        ::Tags::FixedSource<Tags::MMode>, Tags::SingularField,
        ::Tags::deriv<Tags::SingularField, tmpl::size_t<2>, Frame::Inertial>,
        Tags::BoyerLindquistRadius, Tags::RawEffSource,
        Tags::EF_EffSource,Tags::RawPuncture, Tags::EF_Puncture> /*meta*/,
    const bool field_is_regularized) const {
  const double black_hole_spin_ = circular_orbit_.black_hole_spin();
  const double black_hole_mass_ = circular_orbit_.black_hole_mass();
  const int version_ = circular_orbit_.version();
  // const double orbital_radius_ = circular_orbit_.orbital_radius();
  const int m_mode_number_ = circular_orbit_.m_mode_number();
  const double a = black_hole_spin_ * black_hole_mass_;
  const double M = black_hole_mass_;
  // const double r_0 = orbital_radius_;
  const double r_plus = M * (1. + sqrt(1. - square(black_hole_spin_)));
  const double r_minus = M * (1. - sqrt(1. - square(black_hole_spin_)));
  const auto& r = get<0>(x);
  const DataVector theta =
      acos(get<1>(x));  // get<1>(x) is cos_theta when penetrating_horizon
  const DataVector r_minus_r_plus = r - r_plus;
  const DataVector delta = r_minus_r_plus * (r - r_minus);
  const DataVector delta_phi = m_mode_number_ * a / (r_plus - r_minus) *
                               log((r - r_plus) / (r - r_minus));
  const ComplexDataVector rotation =
      cos(delta_phi) + std::complex<double>(0., 1.) * sin(delta_phi);
  tuples::TaggedTuple<
      ::Tags::FixedSource<Tags::MMode>, Tags::SingularField,
      ::Tags::deriv<Tags::SingularField, tmpl::size_t<2>, Frame::Inertial>,
      Tags::BoyerLindquistRadius, Tags::RawEffSource, Tags::EF_EffSource,
      Tags::RawPuncture, Tags::EF_Puncture>
      result{};
  get(get<Tags::BoyerLindquistRadius>(result)) = r;
  const size_t num_points = get<0>(x).size();
  tnsr::aa<ComplexDataVector, 3>& effective_source =
      get<::Tags::FixedSource<Tags::MMode>>(result);
  tnsr::aa<ComplexDataVector, 3>& singular_field =
      get<Tags::SingularField>(result);
  tnsr::aa<ComplexDataVector, 3>& raw_eff_source =
      get<Tags::RawEffSource>(result);
  tnsr::aa<ComplexDataVector, 3>& ef_eff_source =
      get<Tags::EF_EffSource>(result);
  tnsr::aa<ComplexDataVector, 3>& raw_puncture =
      get<Tags::RawPuncture>(result);
  tnsr::aa<ComplexDataVector, 3>& ef_puncture =
      get<Tags::EF_Puncture>(result);
  for (size_t i = 0; i < singular_field.size(); i++) {
    effective_source[i].destructive_resize(num_points);
    singular_field[i].destructive_resize(num_points);
    raw_eff_source[i].destructive_resize(num_points);
    ef_eff_source[i].destructive_resize(num_points);
    raw_puncture[i].destructive_resize(num_points);
    ef_puncture[i].destructive_resize(num_points);
  }
  auto& deriv_singular_field =
      get<::Tags::deriv<Tags::SingularField, tmpl::size_t<2>, Frame::Inertial>>(
          result);
  for (size_t i = 0; i < deriv_singular_field.size(); i++) {
    deriv_singular_field[i].destructive_resize(num_points);
  }
  // Decide which interpolator to use based on position
  const auto& hyperboloidal_slicing_transitions =
      circular_orbit_.hyperboloidal_slicing_transitions().value();
  const auto& interpolator = [&]() {
    if (field_is_regularized) {
      return interpolators_[3].interpolator;
    }
    const double any_r = r[0];
    if (any_r < hyperboloidal_slicing_transitions[0]) {
      return interpolators_[0].interpolator;
    } else if (any_r < hyperboloidal_slicing_transitions[2]) {
      return interpolators_[1].interpolator;
    } else {
      return interpolators_[2].interpolator;
    }
  }();
  const std::array<std::array<double, 2>, 2> interpolator_bounds{
      {{{interpolator.lower_bound(0), interpolator.upper_bound(0)}},
       {{interpolator.lower_bound(1), interpolator.upper_bound(1)}}}};
  // Interpolate data
  // Ordering of components:
  // tt, tr, ttheta, tphi, rr, rtheta, rphi, theta theta, theta phi, phi phi
  for (size_t i = 0; i < num_points; ++i) {
    const double r_clamped =
        std::clamp(r[i], interpolator_bounds[0][0], interpolator_bounds[0][1]);
    if (not equal_within_roundoff(r[i], r_clamped)) {
      ERROR("Requested r = " << r[i] << " outside of interpolation bounds ["
                             << interpolator_bounds[0][0] << ", "
                             << interpolator_bounds[0][1] << "]");
    }
    const double theta_clamped = std::clamp(theta[i], interpolator_bounds[1][0],
                                            interpolator_bounds[1][1]);
    if ((theta[i] < interpolator_bounds[1][0] or
         theta[i] > interpolator_bounds[1][1]) and
        // Allow extrapolation to the poles
        not(equal_within_roundoff(theta[i], 0., 0.15) or
            equal_within_roundoff(theta[i], M_PI, 0.15) or
            equal_within_roundoff(theta[i], interpolator_bounds[1][0]) or
            equal_within_roundoff(theta[i], interpolator_bounds[1][1]))) {
      ERROR("Requested theta = " << theta[i]
                                 << " outside of interpolation bounds ["
                                 << interpolator_bounds[1][0] << ", "
                                 << interpolator_bounds[1][1] << "]");
    }
    const auto weights = interpolator.get_weights(r_clamped, theta_clamped);
    // Load raw BL-frame source from h5 (upper-triangular ordering: k=0..9)
    std::array<double, 10> src_re_arr{};
    std::array<double, 10> src_im_arr{};
    std::array<double, 10> src_conv_re{};
    std::array<double, 10> src_conv_im{};
    for (size_t k = 0; k < 10; ++k) {
      gsl::at(src_re_arr, k) = interpolator.interpolate(weights, 2 * k);
      gsl::at(src_im_arr, k) = interpolator.interpolate(weights, 2 * k + 1);
      if (pi_2_rotation_) {
        const std::complex<double> rotated =
            (gsl::at(src_re_arr, k) +
             std::complex<double>(0., 1.) * gsl::at(src_im_arr, k)) *
            (2. * M_PI * rotation[i]);
        gsl::at(src_re_arr, k) = rotated.real();
        gsl::at(src_im_arr, k) = rotated.imag();
      }
    }
    if (version_ == 2) {
      // Convert from BL frame to VR (comoving ingoing EF) frame
      detail::convert_effsource_Seff_vr(m_mode_number_, a, r[i], get<1>(x)[i],
                                        src_re_arr, src_im_arr, src_conv_re,
                                        src_conv_im);
    } else if (version_ == 3) {
      // Convert from BL frame to VR (comoving ingoing EF) frame
      detail::convert_effsource_Seff_vrz(m_mode_number_, a, r[i], get<1>(x)[i],
                                         src_re_arr, src_im_arr, src_conv_re,
                                         src_conv_im);
    }

    // Store into SpECTRE lower-triangular ordering
    for (size_t a1 = 0; a1 < 4; ++a1) {
      for (size_t b = 0; b <= a1; ++b) {
        const size_t comp = tnsr::aa<ComplexDataVector, 3>::get_storage_index(
            std::array<size_t, 2>{{a1, b}});
        // effective_source.get(a1, b)[i] =
        //     gsl::at(src_conv_re, comp) +
        //     std::complex<double>(0., 1.) * gsl::at(src_conv_im, comp);

        // CAUTION: In CircularOrbit, we flip sign when penetrating horizon.
        // Do we need to do that here as well?
        effective_source.get(a1, b)[i] =
            -gsl::at(src_conv_re, comp) -
            std::complex<double>(0., 1.) * gsl::at(src_conv_im, comp);
        raw_eff_source.get(a1, b)[i] =
            gsl::at(src_re_arr, comp) +
            std::complex<double>(0., 1.) * gsl::at(src_im_arr, comp);
        ef_eff_source.get(a1, b)[i] =
            gsl::at(src_conv_re, comp) +
            std::complex<double>(0., 1.) * gsl::at(src_conv_im, comp);
      }
    }


    // Load the interior puncture (singular) field from h5 (upper-triangular
    // ordering: k=0..9), apply pi_2_rotation, and convert from BL to EF
    // frame with the same psi conversion used for the boundary hS data.
    // Only defined in the regularized region, where the Puncture dataset
    // shares its (r, theta) grid with Seff.
    if (field_is_regularized) {
      const auto puncture_weights =
          interpolators_[4].interpolator.get_weights(r_clamped, theta_clamped);
      std::array<double, 10> hP_re_arr{};
      std::array<double, 10> hP_im_arr{};
      std::array<double, 10> hP_conv_re{};
      std::array<double, 10> hP_conv_im{};
      for (size_t k = 0; k < 10; ++k) {
        gsl::at(hP_re_arr, k) =
            interpolators_[4].interpolator.interpolate(puncture_weights, 2 * k);
        gsl::at(hP_im_arr, k) = interpolators_[4].interpolator.interpolate(
            puncture_weights, 2 * k + 1);
        if (pi_2_rotation_) {
          const std::complex<double> rotated =
              (gsl::at(hP_re_arr, k) +
               std::complex<double>(0., 1.) * gsl::at(hP_im_arr, k)) *
              (2. * M_PI * rotation[i]);
          gsl::at(hP_re_arr, k) = rotated.real();
          gsl::at(hP_im_arr, k) = rotated.imag();
        }
      }
      if (version_ == 2) {
        detail::convert_effsource_psi_vr(m_mode_number_, a, r[i], get<1>(x)[i],
                                         hP_re_arr, hP_im_arr, hP_conv_re,
                                         hP_conv_im);
      } else if (version_ == 3) {
        detail::convert_effsource_psi_vrz(m_mode_number_, a, r[i], get<1>(x)[i],
                                          hP_re_arr, hP_im_arr, hP_conv_re,
                                          hP_conv_im);
      }
      for (size_t a1 = 0; a1 < 4; ++a1) {
        for (size_t b = 0; b <= a1; ++b) {
          const size_t comp = tnsr::aa<ComplexDataVector, 3>::get_storage_index(
              std::array<size_t, 2>{{a1, b}});
          raw_puncture.get(a1, b)[i] =
              gsl::at(hP_re_arr, comp) +
              std::complex<double>(0., 1.) * gsl::at(hP_im_arr, comp);
          ef_puncture.get(a1, b)[i] =
              gsl::at(hP_conv_re, comp) +
              std::complex<double>(0., 1.) * gsl::at(hP_conv_im, comp);
        }
      }
    } else {
      for (size_t a1 = 0; a1 < 4; ++a1) {
        for (size_t b = 0; b <= a1; ++b) {
          raw_puncture.get(a1, b)[i] = 0.;
          ef_puncture.get(a1, b)[i] = 0.;
        }
      }
    }

  }

  {
    // Puncture, drPuncture, dthPuncture are provided wherever the
    // corresponding "true" effective source (Seff inside the regularized
    // block, RetRetV/T/U outside it) is nonzero, i.e. at least the T-slicing
    // region. Populate singular_field and deriv_singular_field from them at
    // every point within their footprint, regardless of field_is_regularized:
    // when field_is_regularized is true the point must always be covered (it
    // lies in the regularized block), so we error loudly if it isn't; when
    // false, points may legitimately fall outside the puncture footprint
    // (e.g. deep in the RetRetU region), so we just leave the field zero
    // there. (Downstream, InitializeEffectiveSource only reads these tags
    // when field_is_regularized is true and independently zeroes them itself
    // otherwise, so populating them further out here doesn't affect the
    // actual solve.)
    for (auto& component : singular_field) {
      component = ComplexDataVector(num_points, 0.0);
    }
    for (auto& component : deriv_singular_field) {
      component = ComplexDataVector(num_points, 0.0);
    }
    const auto& puncture_interpolator = interpolators_[4].interpolator;
    const auto& dr_puncture_interpolator = interpolators_[5].interpolator;
    const auto& dth_puncture_interpolator = interpolators_[6].interpolator;
    const std::array<std::array<double, 2>, 2> puncture_bounds{
        {{{puncture_interpolator.lower_bound(0),
           puncture_interpolator.upper_bound(0)}},
         {{puncture_interpolator.lower_bound(1),
           puncture_interpolator.upper_bound(1)}}}};
    for (size_t i = 0; i < num_points; ++i) {
      const double r_clamped = std::clamp(r[i], puncture_bounds[0][0],
                                          puncture_bounds[0][1]);
      const bool r_out_of_bounds = not equal_within_roundoff(r[i], r_clamped);
      const double theta_clamped = std::clamp(theta[i], puncture_bounds[1][0],
                                              puncture_bounds[1][1]);
      const bool theta_out_of_bounds =
          (theta[i] < puncture_bounds[1][0] or
           theta[i] > puncture_bounds[1][1]) and
          // Allow extrapolation to the poles
          not(equal_within_roundoff(theta[i], 0., 0.15) or
              equal_within_roundoff(theta[i], M_PI, 0.15) or
              equal_within_roundoff(theta[i], puncture_bounds[1][0]) or
              equal_within_roundoff(theta[i], puncture_bounds[1][1]));
      if (r_out_of_bounds or theta_out_of_bounds) {
        if (field_is_regularized) {
          ERROR("Requested (r, theta) = ("
                << r[i] << ", " << theta[i]
                << ") is outside of the puncture interpolation bounds "
                   "[r: "
                << puncture_bounds[0][0] << ", " << puncture_bounds[0][1]
                << "], [theta: " << puncture_bounds[1][0] << ", "
                << puncture_bounds[1][1]
                << "], but field_is_regularized is true.");
        }
        // Outside the puncture footprint and not regularized here: leave
        // singular_field / deriv_singular_field zero at this point.
        continue;
      }
      const auto weights =
          puncture_interpolator.get_weights(r_clamped, theta_clamped);
      const auto dr_weights =
          dr_puncture_interpolator.get_weights(r_clamped, theta_clamped);
      const auto dth_weights =
          dth_puncture_interpolator.get_weights(r_clamped, theta_clamped);

      // Load raw BL-frame hS and its r- and theta-derivatives from h5
      // (upper-triangular ordering k=0..9) and convert to VR frame.
      // NOLINTBEGIN(cppcoreguidelines-pro-bounds-constant-array-index)
      std::array<double, 10> hS_re_arr{};
      std::array<double, 10> hS_im_arr{};
      std::array<double, 10> dhS_dr_re_arr{};
      std::array<double, 10> dhS_dr_im_arr{};
      std::array<double, 10> dhS_dth_re_arr{};
      std::array<double, 10> dhS_dth_im_arr{};
      std::array<double, 10> hS_conv_re{};
      std::array<double, 10> hS_conv_im{};
      std::array<double, 10> dhS_dr_conv_re{};
      std::array<double, 10> dhS_dr_conv_im{};
      std::array<double, 10> dhS_dth_conv_re{};
      std::array<double, 10> dhS_dth_conv_im{};
      for (size_t k = 0; k < 10; ++k) {
        hS_re_arr[k] = puncture_interpolator.interpolate(weights, 2 * k);
        hS_im_arr[k] = puncture_interpolator.interpolate(weights, 2 * k + 1);
        dhS_dr_re_arr[k] =
            dr_puncture_interpolator.interpolate(dr_weights, 2 * k);
        dhS_dr_im_arr[k] =
            dr_puncture_interpolator.interpolate(dr_weights, 2 * k + 1);
        dhS_dth_re_arr[k] =
            dth_puncture_interpolator.interpolate(dth_weights, 2 * k);
        dhS_dth_im_arr[k] =
            dth_puncture_interpolator.interpolate(dth_weights, 2 * k + 1);
        if (pi_2_rotation_) {
          const std::complex<double> rotated_hS =
              (hS_re_arr[k] + std::complex<double>(0., 1.) * hS_im_arr[k]) *
              (2. * M_PI * rotation[i]);
          hS_re_arr[k] = rotated_hS.real();
          hS_im_arr[k] = rotated_hS.imag();
          const std::complex<double> rotated_dhS_dr =
              2. * M_PI * rotation[i] *
              (dhS_dr_re_arr[k] +
               std::complex<double>(0., 1.) * dhS_dr_im_arr[k]);
          dhS_dr_re_arr[k] = rotated_dhS_dr.real();
          dhS_dr_im_arr[k] = rotated_dhS_dr.imag();
          const std::complex<double> rotated_dhS_dth =
              2. * M_PI * rotation[i] *
              (dhS_dth_re_arr[k] +
               std::complex<double>(0., 1.) * dhS_dth_im_arr[k]);
          dhS_dth_re_arr[k] = rotated_dhS_dth.real();
          dhS_dth_im_arr[k] = rotated_dhS_dth.imag();
        }
      }
      if (version_ == 2) {
        detail::convert_effsource_psi_vr(m_mode_number_, a, r[i], get<1>(x)[i],
                                         hS_re_arr, hS_im_arr, hS_conv_re,
                                         hS_conv_im);
        detail::convert_effsource_dpsidr_vr(
            m_mode_number_, a, r[i], get<1>(x)[i], hS_re_arr, hS_im_arr,
            dhS_dr_re_arr, dhS_dr_im_arr, dhS_dr_conv_re, dhS_dr_conv_im);
        detail::convert_effsource_dpsidz_vr(
            m_mode_number_, a, r[i], get<1>(x)[i], hS_re_arr, hS_im_arr,
            dhS_dth_re_arr, dhS_dth_im_arr, dhS_dth_conv_re, dhS_dth_conv_im);
      } else if (version_ == 3) {
        detail::convert_effsource_psi_vrz(m_mode_number_, a, r[i], get<1>(x)[i],
                                          hS_re_arr, hS_im_arr, hS_conv_re,
                                          hS_conv_im);
        detail::convert_effsource_dpsidr_vrz(
            m_mode_number_, a, r[i], get<1>(x)[i], hS_re_arr, hS_im_arr,
            dhS_dr_re_arr, dhS_dr_im_arr, dhS_dr_conv_re, dhS_dr_conv_im);
        detail::convert_effsource_dpsidz_vrz(
            m_mode_number_, a, r[i], get<1>(x)[i], hS_re_arr, hS_im_arr,
            dhS_dth_re_arr, dhS_dth_im_arr, dhS_dth_conv_re, dhS_dth_conv_im);
      }
      for (size_t a1 = 0; a1 < 4; ++a1) {
        for (size_t b = 0; b <= a1; ++b) {
          const size_t comp = tnsr::aa<ComplexDataVector, 3>::get_storage_index(
              std::array<size_t, 2>{{a1, b}});
          singular_field.get(a1, b)[i] =
              hS_conv_re[comp] +
              std::complex<double>(0., 1.) * hS_conv_im[comp];
          deriv_singular_field.get(0, a1, b)[i] =
              dhS_dr_conv_re[comp] +
              std::complex<double>(0., 1.) * dhS_dr_conv_im[comp];
          deriv_singular_field.get(1, a1, b)[i] =
              dhS_dth_conv_re[comp] +
              std::complex<double>(0., 1.) * dhS_dth_conv_im[comp];
        }
      }
      // NOLINTEND(cppcoreguidelines-pro-bounds-constant-array-index)
    }
  }

  return result;
}

// Fixed sources, plus diagnostic-only tags (e.g. Tags::RHSBoxPuncture). This
// computes the same six tags as the overload above by calling it directly
// (no added cost for the actual solve, which only ever calls that overload),
// then adds the diagnostic-only computation on top.
tuples::TaggedTuple<
    ::Tags::FixedSource<Tags::MMode>, Tags::SingularField,
    ::Tags::deriv<Tags::SingularField, tmpl::size_t<2>, Frame::Inertial>,
    Tags::BoyerLindquistRadius, Tags::RawEffSource, Tags::EF_EffSource,
    Tags::RawPuncture, Tags::EF_Puncture,Tags::RHSBoxPuncture>
NumericData::variables(
    const tnsr::I<DataVector, 2>& x,
    tmpl::list<
        ::Tags::FixedSource<Tags::MMode>, Tags::SingularField,
        ::Tags::deriv<Tags::SingularField, tmpl::size_t<2>, Frame::Inertial>,
        Tags::BoyerLindquistRadius, Tags::RawEffSource, Tags::EF_EffSource,
        Tags::RawPuncture, Tags::EF_Puncture,Tags::RHSBoxPuncture> /*meta*/,
    const bool field_is_regularized) const {
  const auto base = variables(x, source_tags{}, field_is_regularized);
  tuples::TaggedTuple<
      ::Tags::FixedSource<Tags::MMode>, Tags::SingularField,
      ::Tags::deriv<Tags::SingularField, tmpl::size_t<2>, Frame::Inertial>,
      Tags::BoyerLindquistRadius, Tags::RawEffSource, Tags::EF_EffSource,
      Tags::RawPuncture, Tags::EF_Puncture,Tags::RHSBoxPuncture>
      result{};
  get<::Tags::FixedSource<Tags::MMode>>(result) =
      get<::Tags::FixedSource<Tags::MMode>>(base);
  get<Tags::SingularField>(result) = get<Tags::SingularField>(base);
  get<::Tags::deriv<Tags::SingularField, tmpl::size_t<2>, Frame::Inertial>>(
      result) =
      get<::Tags::deriv<Tags::SingularField, tmpl::size_t<2>, Frame::Inertial>>(
          base);
  get<Tags::BoyerLindquistRadius>(result) =
      get<Tags::BoyerLindquistRadius>(base);
  get<Tags::RawEffSource>(result) = get<Tags::RawEffSource>(base);
  get<Tags::EF_EffSource>(result) = get<Tags::EF_EffSource>(base);
  get<Tags::RawPuncture>(result) = get<Tags::RawPuncture>(base);
  get<Tags::EF_Puncture>(result) = get<Tags::EF_Puncture>(base);

  const double black_hole_spin_ = circular_orbit_.black_hole_spin();
  const double black_hole_mass_ = circular_orbit_.black_hole_mass();
  const int version_ = circular_orbit_.version();
  const int m_mode_number_ = circular_orbit_.m_mode_number();
  const double a = black_hole_spin_ * black_hole_mass_;
  const double M = black_hole_mass_;
  const double r_plus = M * (1. + sqrt(1. - square(black_hole_spin_)));
  const double r_minus = M * (1. - sqrt(1. - square(black_hole_spin_)));
  const auto& r = get<0>(x);
  const DataVector theta =
      acos(get<1>(x));  // get<1>(x) is cos_theta when penetrating_horizon
  const DataVector delta_phi = m_mode_number_ * a / (r_plus - r_minus) *
                               log((r - r_plus) / (r - r_minus));
  const ComplexDataVector rotation =
      cos(delta_phi) + std::complex<double>(0., 1.) * sin(delta_phi);
  const size_t num_points = get<0>(x).size();

  tnsr::aa<ComplexDataVector, 3>& rhs_box_puncture =
      get<Tags::RHSBoxPuncture>(result);
  for (auto& component : rhs_box_puncture) {
    component = ComplexDataVector(num_points, 0.0);
  }

  // RHSBoxPuncture = Seff - RetRetT (see Tags::RHSBoxPuncture doc comment).
  // Each term uses the same sign-flip convention as effective_source in the
  // overload above, but with a soft out-of-bounds check (contribute zero
  // rather than erroring) since Seff and RetRetT each only cover part of the
  // T-slicing region.
  const std::array<std::pair<size_t, double>, 2> rhs_box_puncture_terms{
      {{3, 1.}, {1, -1.}}};  // {interpolator index, sign}: Seff, RetRetT
  for (const auto& [interp_index, sign] : rhs_box_puncture_terms) {
    const auto& interp = interpolators_[interp_index].interpolator;
    const std::array<std::array<double, 2>, 2> bounds{
        {{{interp.lower_bound(0), interp.upper_bound(0)}},
         {{interp.lower_bound(1), interp.upper_bound(1)}}}};
    for (size_t i = 0; i < num_points; ++i) {
      const double r_clamped = std::clamp(r[i], bounds[0][0], bounds[0][1]);
      if (not equal_within_roundoff(r[i], r_clamped)) {
        continue;  // Outside this dataset's r range
      }
      const double theta_clamped =
          std::clamp(theta[i], bounds[1][0], bounds[1][1]);
      if ((theta[i] < bounds[1][0] or theta[i] > bounds[1][1]) and
          not(equal_within_roundoff(theta[i], 0., 0.15) or
              equal_within_roundoff(theta[i], M_PI, 0.15) or
              equal_within_roundoff(theta[i], bounds[1][0]) or
              equal_within_roundoff(theta[i], bounds[1][1]))) {
        continue;  // Outside this dataset's theta range
      }
      const auto weights = interp.get_weights(r_clamped, theta_clamped);
      // NOLINTBEGIN(cppcoreguidelines-pro-bounds-constant-array-index)
      std::array<double, 10> src_re_arr{};
      std::array<double, 10> src_im_arr{};
      std::array<double, 10> src_conv_re{};
      std::array<double, 10> src_conv_im{};
      for (size_t k = 0; k < 10; ++k) {
        src_re_arr[k] = interp.interpolate(weights, 2 * k);
        src_im_arr[k] = interp.interpolate(weights, 2 * k + 1);
        if (pi_2_rotation_) {
          const std::complex<double> rotated =
              (src_re_arr[k] +
               std::complex<double>(0., 1.) * src_im_arr[k]) *
              (2. * M_PI * rotation[i]);
          src_re_arr[k] = rotated.real();
          src_im_arr[k] = rotated.imag();
        }
      }
      if (version_ == 2) {
        detail::convert_effsource_Seff_vr(m_mode_number_, a, r[i],
                                          get<1>(x)[i], src_re_arr,
                                          src_im_arr, src_conv_re,
                                          src_conv_im);
      } else if (version_ == 3) {
        detail::convert_effsource_Seff_vrz(m_mode_number_, a, r[i],
                                           get<1>(x)[i], src_re_arr,
                                           src_im_arr, src_conv_re,
                                           src_conv_im);
      }
      for (size_t a1 = 0; a1 < 4; ++a1) {
        for (size_t b = 0; b <= a1; ++b) {
          const size_t comp =
              tnsr::aa<ComplexDataVector, 3>::get_storage_index(
                  std::array<size_t, 2>{{a1, b}});
          rhs_box_puncture.get(a1, b)[i] +=
              sign * (-gsl::at(src_conv_re, comp) -
                     std::complex<double>(0., 1.) *
                         gsl::at(src_conv_im, comp));
        }
      }
      // NOLINTEND(cppcoreguidelines-pro-bounds-constant-array-index)
    }
  }

  return result;
}

tuples::TaggedTuple<
    Tags::SingularField,
    ::Tags::deriv<Tags::SingularField, tmpl::size_t<2>, Frame::Inertial>>
NumericData::boundary_face_variables(const tnsr::I<DataVector, 2>& x,
                                     const size_t face) const {
  ASSERT(face < 4,
         "face must be 0 (Left), 1 (Right), 2 (Bottom), or 3 (Top), but got "
             << face);
  // Left/Right (face 0/1): normal is r, parameterized by theta.
  // Bottom/Top (face 2/3): normal is theta, parameterized by r.
  const bool on_r_face = (face == 0 or face == 1);
  const double black_hole_spin_ = circular_orbit_.black_hole_spin();
  const double black_hole_mass_ = circular_orbit_.black_hole_mass();
  const int version_ = circular_orbit_.version();
  const int m_mode_number_ = circular_orbit_.m_mode_number();
  const double a = black_hole_spin_ * black_hole_mass_;
  const double M = black_hole_mass_;
  const double r_plus = M * (1. + sqrt(1. - square(black_hole_spin_)));
  const double r_minus = M * (1. - sqrt(1. - square(black_hole_spin_)));
  const auto& r = get<0>(x);
  const DataVector theta = acos(get<1>(x));
  const DataVector delta_phi = m_mode_number_ * a / (r_plus - r_minus) *
                               log((r - r_plus) / (r - r_minus));
  const ComplexDataVector rotation =
      cos(delta_phi) + std::complex<double>(0., 1.) * sin(delta_phi);

  tuples::TaggedTuple<
      Tags::SingularField,
      ::Tags::deriv<Tags::SingularField, tmpl::size_t<2>, Frame::Inertial>>
      result{};
  const size_t num_points = get<0>(x).size();
  tnsr::aa<ComplexDataVector, 3>& singular_field =
      get<Tags::SingularField>(result);
  auto& deriv_singular_field =
      get<::Tags::deriv<Tags::SingularField, tmpl::size_t<2>, Frame::Inertial>>(
          result);
  for (auto& component : singular_field) {
    component = ComplexDataVector(num_points, 0.0);
  }
  for (auto& component : deriv_singular_field) {
    component = ComplexDataVector(num_points, 0.0);
  }

  const auto& binterp = boundary_interpolators_[face].interpolator;
  for (size_t i = 0; i < num_points; ++i) {
    const double coord_i = on_r_face ? theta[i] : r[i];
    const auto weights = binterp.get_weights(coord_i);

    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-constant-array-index)
    std::array<double, 10> hS_re_arr{};
    std::array<double, 10> hS_im_arr{};
    std::array<double, 10> dhS_re_arr{};
    std::array<double, 10> dhS_im_arr{};
    std::array<double, 10> hS_conv_re{};
    std::array<double, 10> hS_conv_im{};
    std::array<double, 10> dhS_conv_re{};
    std::array<double, 10> dhS_conv_im{};
    for (size_t k = 0; k < 10; ++k) {
      hS_re_arr[k] = binterp.interpolate(weights, 2 * k);
      hS_im_arr[k] = binterp.interpolate(weights, 2 * k + 1);
      dhS_re_arr[k] = binterp.interpolate(weights, 20 + 2 * k);
      dhS_im_arr[k] = binterp.interpolate(weights, 20 + 2 * k + 1);
      if (pi_2_rotation_) {
        const std::complex<double> rotated_hS =
            (hS_re_arr[k] + std::complex<double>(0., 1.) * hS_im_arr[k]) *
            (2. * M_PI * rotation[i]);
        hS_re_arr[k] = rotated_hS.real();
        hS_im_arr[k] = rotated_hS.imag();
        const std::complex<double> rotated_dhS =
            2. * M_PI * rotation[i] *
            (dhS_re_arr[k] + std::complex<double>(0., 1.) * dhS_im_arr[k]);
        dhS_re_arr[k] = rotated_dhS.real();
        dhS_im_arr[k] = rotated_dhS.imag();
      }
    }
    if (version_ == 2) {
      detail::convert_effsource_psi_vr(m_mode_number_, a, r[i], get<1>(x)[i],
                                       hS_re_arr, hS_im_arr, hS_conv_re,
                                       hS_conv_im);
      if (on_r_face) {
        detail::convert_effsource_dpsidr_vr(
            m_mode_number_, a, r[i], get<1>(x)[i], hS_re_arr, hS_im_arr,
            dhS_re_arr, dhS_im_arr, dhS_conv_re, dhS_conv_im);
      } else {
        detail::convert_effsource_dpsidz_vr(
            m_mode_number_, a, r[i], get<1>(x)[i], hS_re_arr, hS_im_arr,
            dhS_re_arr, dhS_im_arr, dhS_conv_re, dhS_conv_im);
      }
    } else if (version_ == 3) {
      detail::convert_effsource_psi_vrz(m_mode_number_, a, r[i], get<1>(x)[i],
                                        hS_re_arr, hS_im_arr, hS_conv_re,
                                        hS_conv_im);
      if (on_r_face) {
        detail::convert_effsource_dpsidr_vrz(
            m_mode_number_, a, r[i], get<1>(x)[i], hS_re_arr, hS_im_arr,
            dhS_re_arr, dhS_im_arr, dhS_conv_re, dhS_conv_im);
      } else {
        detail::convert_effsource_dpsidz_vrz(
            m_mode_number_, a, r[i], get<1>(x)[i], hS_re_arr, hS_im_arr,
            dhS_re_arr, dhS_im_arr, dhS_conv_re, dhS_conv_im);
      }
    }
    for (size_t a1 = 0; a1 < 4; ++a1) {
      for (size_t b = 0; b <= a1; ++b) {
        const size_t comp = tnsr::aa<ComplexDataVector, 3>::get_storage_index(
            std::array<size_t, 2>{{a1, b}});
        singular_field.get(a1, b)[i] =
            hS_conv_re[comp] +
            std::complex<double>(0., 1.) * hS_conv_im[comp];
        if (on_r_face) {
          deriv_singular_field.get(0, a1, b)[i] =
              dhS_conv_re[comp] +
              std::complex<double>(0., 1.) * dhS_conv_im[comp];
        } else {
          deriv_singular_field.get(1, a1, b)[i] =
              dhS_conv_re[comp] +
              std::complex<double>(0., 1.) * dhS_conv_im[comp];
        }
      }
    }
    // NOLINTEND(cppcoreguidelines-pro-bounds-constant-array-index)
  }
  return result;
}

void NumericData::pup(PUP::er& p) {
  elliptic::analytic_data::Background::pup(p);
  elliptic::analytic_data::InitialGuess::pup(p);
  p | filename_;
  p | circular_orbit_;
  p | pi_2_rotation_;
  if (p.isUnpacking()) {
    interpolators_ = load_all_data(filename_);
    boundary_interpolators_ = load_all_boundary_data(filename_);
  }
}

bool operator==(const NumericData& lhs, const NumericData& rhs) {
  return lhs.filename_ == rhs.filename_ and
         lhs.circular_orbit_ == rhs.circular_orbit_ and
         lhs.pi_2_rotation_ == rhs.pi_2_rotation_;
}

bool operator!=(const NumericData& lhs, const NumericData& rhs) {
  return not(lhs == rhs);
}

PUP::able::PUP_ID NumericData::my_PUP_ID = 0;  // NOLINT

}  // namespace GrSelfForce::AnalyticData
