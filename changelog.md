# Changelog

## [0.4.3][2018-05-22]
### Fixed
- c_char instead of i8 in libsvm bindings

## [0.4.2][unreleased]
### Fixed
- fixed EncodableRng dummy serialization implementation

## [0.4.2][unreleased]
### Fixed
- fixed EncodableRng dummy serialization implementation

## [0.4.1][2016-08-26]
### Fixed
- panic when removing constant features when splitting
  a decision tree

## [0.4.0][2016-08-08]
### Added
- factorization machines
- parallel fitting and prediction for one-vs-rest models

## [0.3.1][2016-03-01]
### Changed
- NonzeroIterable now takes &self
- bincode version bumped to 0.4.1 to fix dependency breakage

## [0.3.0][2016-01-16]
### Added
- added ROC AUC score metric

### Changed
- moved newsgroups dataset code into a central module
- made the utils module semi-public

## [0.2.0][2015-12-14]
### Added
- added libsvm bindings and the SVC model.
### Changed
- replace input checking asserts in model functions with error-returning functions

## [0.1.0][2015-12-06]
### Changed
- Initial release.
