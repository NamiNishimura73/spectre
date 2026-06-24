# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY GrSelfForce)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Equations.cpp
  LorenzGaugeConditions.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Equations.hpp
  FirstOrderSystem.hpp
  LorenzGaugeConditions.hpp
  Tags.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  DomainStructure
  ErrorHandling
  GeneralRelativity
  GrSelfForceAnalyticData
  Utilities
  )

add_subdirectory(Actions)
add_subdirectory(AmrCriteria)
add_subdirectory(AnalyticData)
add_subdirectory(BoundaryConditions)
add_subdirectory(Events)
