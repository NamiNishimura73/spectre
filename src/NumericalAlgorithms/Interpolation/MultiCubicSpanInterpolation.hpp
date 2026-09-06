// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>

#include "DataStructures/Index.hpp"
#include "NumericalAlgorithms/Interpolation/LagrangePolynomial.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"

namespace intrp {

/*!
 * \brief Performs bicubic interpolation on a uniform 2D grid, using a 4-point
 * (per dimension) Lagrange stencil, i.e. 16 points total per interpolated
 * value.
 *
 * The class is non-owning and expects a C-ordered array (n, x, y), with the
 * variable index n varying fastest -- the same data layout as
 * MultiLinearSpanInterpolation. Unlike that class, this one is fixed to 2
 * dimensions and always assumes uniform grid spacing in each dimension, since
 * that is the only case currently needed (interpolating source data that is
 * expected to be smooth, as opposed to data with a puncture-like feature
 * where a lower-order, more local scheme is preferred).
 *
 * The 1D weights at each interpolation point are the Lagrange basis
 * polynomials for the 4 nodes bracketing the target point (2 on each side),
 * computed via `lagrange_polynomial`. Points that fall in the outermost
 * source cell on either side of a dimension are interpolated with a one-sided
 * (but still 4-point, still cubic) stencil rather than one centered on their
 * own cell.
 */
template <size_t NumberOfVariables>
class UniformMultiCubicSpanInterpolation {
 public:
  struct Weight {
    std::array<double, 16> weights;
    Index<16> index;
  };

  size_t extents(const size_t which_dimension) const {
    return number_of_points_[which_dimension];
  }

  /// Compute interpolation weights for the 2D table
  Weight get_weights(double x1, double x2) const;

  double interpolate(const Weight& weights,
                     const size_t which_variable = 0) const {
    double result = 0.;
    for (size_t nn = 0; nn < weights.weights.size(); ++nn) {
      result += weights.weights[nn] *
                y_[which_variable + NumberOfVariables * weights.index[nn]];
    }
    return result;
  }

  UniformMultiCubicSpanInterpolation() = default;

  UniformMultiCubicSpanInterpolation(std::array<gsl::span<const double>, 2> x,
                                     gsl::span<const double> y,
                                     Index<2> number_of_points);

  double lower_bound(const size_t which_dimension) const {
    return x_[which_dimension][0];
  }
  double upper_bound(const size_t which_dimension) const {
    return x_[which_dimension][number_of_points_[which_dimension] - 1];
  }

 private:
  /// Inverse spacing of the table, used to locate the interpolation stencil
  std::array<double, 2> inverse_spacing_{};
  /// Number of points per dimension
  Index<2> number_of_points_{};

  using DataPointer = gsl::span<const double>;
  /// X values of the table
  std::array<DataPointer, 2> x_{};
  /// Y values of the table
  DataPointer y_{};

  /// Base index `i` such that the 4-point stencil {i-1, i, i+1, i+2} is a
  /// valid index range, i.e. `i` in [1, number_of_points - 3]. Assumes
  /// `target` is within (or very close to) the table bounds; the caller is
  /// expected to have already clamped it there, as for
  /// MultiLinearSpanInterpolation.
  size_t stencil_base_index(size_t which_dimension, double target) const;
};

template <size_t NumberOfVariables>
size_t UniformMultiCubicSpanInterpolation<NumberOfVariables>::
    stencil_base_index(const size_t which_dimension,
                       const double target) const {
  const size_t n = number_of_points_[which_dimension];
  ASSERT(n >= 4, "UniformMultiCubicSpanInterpolation needs at least 4 points "
                    "per dimension, got "
                        << n << " in dimension " << which_dimension);
  const double relative =
      (target - x_[which_dimension][0]) * inverse_spacing_[which_dimension];
  // Bracketing index, same convention as MultiLinearSpanInterpolation's
  // find_index_uniform: x[index] <= target < x[index + 1].
  auto index = static_cast<size_t>(std::max(0., relative));
  index = std::min(index, n - 2);
  // Clamp further so the 4-point stencil {index-1, ..., index+2} stays in
  // bounds. Points in the outermost source cell on either side then get a
  // one-sided stencil instead of one centered on their own cell.
  index = std::max(index, size_t{1});
  index = std::min(index, n - 3);
  return index;
}

template <size_t NumberOfVariables>
auto UniformMultiCubicSpanInterpolation<NumberOfVariables>::get_weights(
    const double x1, const double x2) const -> Weight {
  Weight result;
  const size_t i0 = stencil_base_index(0, x1);
  const size_t j0 = stencil_base_index(1, x2);

  const std::array<double, 4> r_stencil{
      {x_[0][i0 - 1], x_[0][i0], x_[0][i0 + 1], x_[0][i0 + 2]}};
  const std::array<double, 4> theta_stencil{
      {x_[1][j0 - 1], x_[1][j0], x_[1][j0 + 1], x_[1][j0 + 2]}};

  std::array<double, 4> wx{};
  std::array<double, 4> wy{};
  for (size_t k = 0; k < 4; ++k) {
    wx[k] = lagrange_polynomial(k, x1, r_stencil.begin(), r_stencil.end());
    wy[k] =
        lagrange_polynomial(k, x2, theta_stencil.begin(), theta_stencil.end());
  }

  // Note: first index (r) varies fastest, matching MultiLinearSpanInterpolation
  for (size_t b = 0; b < 4; ++b) {
    for (size_t a = 0; a < 4; ++a) {
      result.weights[a + 4 * b] = wx[a] * wy[b];
      const auto tmp_index = Index<2>(i0 - 1 + a, j0 - 1 + b);
      result.index[a + 4 * b] = collapsed_index(tmp_index, number_of_points_);
    }
  }
  return result;
}

template <size_t NumberOfVariables>
UniformMultiCubicSpanInterpolation<NumberOfVariables>::
    UniformMultiCubicSpanInterpolation(
        std::array<gsl::span<const double>, 2> x, gsl::span<const double> y,
        Index<2> number_of_points)
    : number_of_points_(number_of_points), x_(x), y_(y) {
  for (size_t i = 0; i < 2; ++i) {
    ASSERT(number_of_points_[i] >= 4,
           "UniformMultiCubicSpanInterpolation needs at least 4 points per "
           "dimension, got "
               << number_of_points_[i] << " in dimension " << i);
    inverse_spacing_[i] = 1. / (x_[i][1] - x_[i][0]);
  }
}

}  // namespace intrp
