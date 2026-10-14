// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/AnalyticData/CircularOrbit.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/AnalyticData/NumericData.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/Tags.hpp"
#include "Elliptic/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace GrSelfForce {

using DerivMMode =
    typename ::Tags::deriv<GrSelfForce::Tags::MMode, tmpl::size_t<2>,
                           Frame::Inertial>::type;

void lorenz_gauge_condition(
    gsl::not_null<tnsr::a<ComplexDataVector, 3>*> result,
    const elliptic::analytic_data::Background& background,
    const tnsr::aa<ComplexDataVector, 3>& field, const DerivMMode& deriv_field,
    const tnsr::I<DataVector, 2, Frame::Inertial>& x,
    bool zero_out);

namespace Tags {

struct LorenzGaugeCondition : db::SimpleTag {
  using type = tnsr::a<ComplexDataVector, 3>;
};

struct LorenzGaugeConditionCompute : LorenzGaugeCondition, db::ComputeTag {
  using base = LorenzGaugeCondition;
  using return_type = tnsr::a<ComplexDataVector, 3>;
  using argument_tags = tmpl::list<
      elliptic::Tags::Background<elliptic::analytic_data::Background>,
      GrSelfForce::Tags::MMode,
      ::Tags::deriv<GrSelfForce::Tags::MMode, tmpl::size_t<2>, Frame::Inertial>,
      domain::Tags::Coordinates<2, Frame::Inertial>,
      GrSelfForce::Tags::FieldIsRegularized>;
  static constexpr auto function = &GrSelfForce::lorenz_gauge_condition;
};

struct LorenzGaugeConditionVSlicing : db::SimpleTag {
  using type = tnsr::a<ComplexDataVector, 3>;
};

struct LorenzGaugeConditionVSlicingCompute : LorenzGaugeConditionVSlicing,
                                             db::ComputeTag {
  using base = LorenzGaugeConditionVSlicing;
  using return_type = tnsr::a<ComplexDataVector, 3>;
  using argument_tags = tmpl::list<
      elliptic::Tags::Background<elliptic::analytic_data::Background>,
      GrSelfForce::Tags::MMode,
      ::Tags::deriv<GrSelfForce::Tags::MMode, tmpl::size_t<2>, Frame::Inertial>,
      domain::Tags::Coordinates<2, Frame::Inertial>,
      GrSelfForce::Tags::FieldIsInVSlicingRegion>;
  static void function(
      gsl::not_null<tnsr::a<ComplexDataVector, 3>*> result,
      const elliptic::analytic_data::Background& background,
      const tnsr::aa<ComplexDataVector, 3>& field, const DerivMMode& deriv_field,
      const tnsr::I<DataVector, 2, Frame::Inertial>& x,
      bool field_is_in_v_slicing_region) {
    lorenz_gauge_condition(result, background, field, deriv_field, x,
                           not field_is_in_v_slicing_region);
  }
};

struct LorenzGaugeConditionTSlicing : db::SimpleTag {
  using type = tnsr::a<ComplexDataVector, 3>;
};

struct LorenzGaugeConditionTSlicingCompute : LorenzGaugeConditionTSlicing,
                                             db::ComputeTag {
  using base = LorenzGaugeConditionTSlicing;
  using return_type = tnsr::a<ComplexDataVector, 3>;
  using argument_tags = tmpl::list<
      elliptic::Tags::Background<elliptic::analytic_data::Background>,
      GrSelfForce::Tags::MMode,
      ::Tags::deriv<GrSelfForce::Tags::MMode, tmpl::size_t<2>, Frame::Inertial>,
      domain::Tags::Coordinates<2, Frame::Inertial>,
      GrSelfForce::Tags::FieldIsInTSlicingRegion,
      GrSelfForce::Tags::FieldIsRegularized>;
  static void function(
      gsl::not_null<tnsr::a<ComplexDataVector, 3>*> result,
      const elliptic::analytic_data::Background& background,
      const tnsr::aa<ComplexDataVector, 3>& field, const DerivMMode& deriv_field,
      const tnsr::I<DataVector, 2, Frame::Inertial>& x,
      bool field_is_in_t_slicing_region, 
      bool field_is_regularized) {
      lorenz_gauge_condition(result, background, field, deriv_field, x,
                            not field_is_in_t_slicing_region or field_is_regularized);

  }
};

struct LorenzGaugeConditionUSlicing : db::SimpleTag {
  using type = tnsr::a<ComplexDataVector, 3>;
};

struct LorenzGaugeConditionUSlicingCompute : LorenzGaugeConditionUSlicing,
                                             db::ComputeTag {
  using base = LorenzGaugeConditionUSlicing;
  using return_type = tnsr::a<ComplexDataVector, 3>;
  using argument_tags = tmpl::list<
      elliptic::Tags::Background<elliptic::analytic_data::Background>,
      GrSelfForce::Tags::MMode,
      ::Tags::deriv<GrSelfForce::Tags::MMode, tmpl::size_t<2>, Frame::Inertial>,
      domain::Tags::Coordinates<2, Frame::Inertial>,
      GrSelfForce::Tags::FieldIsInUSlicingRegion>;
  static void function(
      gsl::not_null<tnsr::a<ComplexDataVector, 3>*> result,
      const elliptic::analytic_data::Background& background,
      const tnsr::aa<ComplexDataVector, 3>& field, const DerivMMode& deriv_field,
      const tnsr::I<DataVector, 2, Frame::Inertial>& x,
      bool field_is_in_u_slicing_region) {
    lorenz_gauge_condition(result, background, field, deriv_field, x,
                           not field_is_in_u_slicing_region);
  }
};

}  // namespace Tags
}  // namespace GrSelfForce
