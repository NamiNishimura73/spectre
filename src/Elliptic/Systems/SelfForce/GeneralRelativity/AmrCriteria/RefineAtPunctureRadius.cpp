// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/SelfForce/GeneralRelativity/AmrCriteria/RefineAtPunctureRadius.hpp"

#include <array>
#include <cmath>
#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Amr/Flag.hpp"
#include "Domain/Block.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Structure/Side.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/AnalyticData/CircularOrbit.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/AnalyticData/NumericData.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/MakeArray.hpp"

namespace GrSelfForce::AmrCriteria {

std::array<amr::Flag, 2> RefineAtPunctureRadius::impl(
    const elliptic::analytic_data::Background& background,
    const Domain<2>& domain, const ElementId<2>& element_id) {
  const auto* co_ptr =
      dynamic_cast<const GrSelfForce::AnalyticData::CircularOrbit*>(
          &background);
  const auto* nd_ptr =
      co_ptr != nullptr
          ? nullptr
          : dynamic_cast<const GrSelfForce::AnalyticData::NumericData*>(
                &background);
  if (co_ptr == nullptr and nd_ptr == nullptr) {
    ERROR("Background must be CircularOrbit or NumericData");
  }
  const auto puncture_position =
      co_ptr != nullptr ? co_ptr->puncture_position()
                        : nd_ptr->puncture_position();
  const auto& block = domain.blocks()[element_id.block_id()];
  // Test point: puncture radius at the angle of this block's center, so that
  // blocks at all angles are handled
  const auto block_center = block.stationary_map()(
      tnsr::I<double, 2, Frame::BlockLogical>{{{0., 0.}}});
  const tnsr::I<double, 2> test_point{
      {{get<0>(puncture_position), get<1>(block_center)}}};
  const auto block_logical_coords =
      block_logical_coordinates_single_point(test_point, block);
  if (not block_logical_coords.has_value()) {
    return make_array<2>(amr::Flag::DoNothing);
  }
  double xi_puncture = get<0>(*block_logical_coords);
  if (std::abs(xi_puncture) < 1e-10) {
    xi_puncture = 0.;
  }
  // Split radially if the puncture radius lies within or on the boundary of
  // this element's radial segment
  const auto& radial_segment = element_id.segment_id(0);
  constexpr double tolerance = 1e-10;
  if (xi_puncture < radial_segment.endpoint(Side::Lower) - tolerance or
      xi_puncture > radial_segment.endpoint(Side::Upper) + tolerance) {
    return make_array<2>(amr::Flag::DoNothing);
  }
  return {{amr::Flag::Split, amr::Flag::DoNothing}};
}

PUP::able::PUP_ID RefineAtPunctureRadius::my_PUP_ID = 0;  // NOLINT

}  // namespace GrSelfForce::AmrCriteria
