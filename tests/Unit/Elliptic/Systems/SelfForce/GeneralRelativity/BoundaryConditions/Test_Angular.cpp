// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <string>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/BoundaryConditions/ApplyBoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryConditionType.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/BoundaryConditions/Angular.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/BoundaryConditions/Factory.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/Pypp.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace GrSelfForce::BoundaryConditions {

SPECTRE_TEST_CASE("Unit.GrSelfForce.BoundaryConditions.Angular",
                  "[Unit][Elliptic]") {
  // Test factory-creation
  const auto created = TestHelpers::test_factory_creation<
      elliptic::BoundaryConditions::BoundaryCondition<2>, Angular>(
      "Angular:\n"
      "  MModeNumber: 3\n");
  REQUIRE(dynamic_cast<const Angular*>(created.get()) != nullptr);
  const auto& boundary_condition = dynamic_cast<const Angular&>(*created);
  {
    INFO("Semantics");
    test_serialization(boundary_condition);
    test_copy_semantics(boundary_condition);
    auto move_boundary_condition = boundary_condition;
    test_move_semantics(std::move(move_boundary_condition), boundary_condition);
  }
  {
    INFO("Properties");
    CHECK(boundary_condition.m_mode_number() == 3);
    CHECK(boundary_condition.boundary_condition_types() ==
          std::vector<elliptic::BoundaryConditionType>{
              elliptic::BoundaryConditionType::Dirichlet});
  }
  {
    INFO("Apply boundary condition");
    const DataVector used_for_size(5);
    auto field = make_with_value<tnsr::aa<ComplexDataVector, 3>>(
        used_for_size, std::numeric_limits<double>::signaling_NaN());
    auto n_dot_field_gradient = make_with_value<tnsr::aa<ComplexDataVector, 3>>(
        used_for_size, std::numeric_limits<double>::signaling_NaN());
    using GradTensorType = TensorMetafunctions::prepend_spatial_index<
        tnsr::aa<ComplexDataVector, 3>, 2, UpLo::Lo, Frame::Inertial>;
    auto deriv_field = make_with_value<GradTensorType>(
        used_for_size, std::numeric_limits<double>::signaling_NaN());
    DirectionMap<2, tnsr::I<ComplexDataVector, 2>> alpha_map{
        {Direction<2>::lower_xi(),
         make_with_value<tnsr::I<ComplexDataVector, 2>>(used_for_size, 0.)}};
    DirectionMap<2, tnsr::aaBB<ComplexDataVector, 3>> beta_map{
        {Direction<2>::lower_xi(),
         make_with_value<tnsr::aaBB<ComplexDataVector, 3>>(used_for_size, 0.)}};
    DirectionMap<2, tnsr::aaBB<ComplexDataVector, 3>> gammarstar_map{
        {Direction<2>::lower_xi(),
         make_with_value<tnsr::aaBB<ComplexDataVector, 3>>(used_for_size, 0.)}};

    const auto box =
        db::create<db::AddSimpleTags<domain::Tags::Faces<2, Tags::Alpha>,
                                     domain::Tags::Faces<2, Tags::Beta>,
                                     domain::Tags::Faces<2, Tags::GammaRstar>>>(
            alpha_map, beta_map, gammarstar_map);
    elliptic::apply_boundary_condition<false, void,
                                       standard_boundary_conditions>(
        boundary_condition, box, Direction<2>::lower_xi(),
        make_not_null(&field), make_not_null(&n_dot_field_gradient),
        deriv_field);
    auto expected_value =
        make_with_value<tnsr::aa<ComplexDataVector, 3>>(used_for_size, 0.);
    CHECK_ITERABLE_APPROX(field, expected_value);
  }
}

}  // namespace GrSelfForce::BoundaryConditions
