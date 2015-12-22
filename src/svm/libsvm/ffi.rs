//! Internals of the `libsvm` FFI.
//!
//! Objects whose names start with `Libsvm` are for the most part
//! things that we pass or get directly from `libsvm`, and are highly
//! fragile.
//!
//! Their safe, memory-owning counterparts start with `Svm`.

use std::slice;
use std::ffi::CStr;

use prelude::*;

/// SVM type.
#[repr(C)]
#[derive(Clone, Debug)]
#[derive(RustcEncodable, RustcDecodable)]
pub enum SvmType {
    C_SVC,
    NU_SVC,
    ONE_CLASS,
    EPSILON_SVR,
    NU_SVR,
}


/// Type of the kernel used by the SVM.
#[repr(C)]
#[derive(Clone, Debug)]
#[derive(RustcEncodable, RustcDecodable)]
pub enum KernelType {
    Linear,
    Polynomial,
    RBF,
    Sigmoid,
    /// Not implemented.
    Precomputed,
}


/// Libsvm uses a sparse representation of data,
/// where every entry in the training matrix
/// is characterised by a column index and a value.
/// Because this is a safe Rust-like object in itself,
/// it does not have a safe counterpart.
#[repr(C)]
#[derive(Clone, Debug)]
#[derive(RustcEncodable, RustcDecodable)]
pub struct LibsvmNode {
    index: i32,
    value: f64,
}


impl LibsvmNode {
    fn new(index: i32, value: f64) -> LibsvmNode {
        LibsvmNode {
            index: index,
            value: value,
        }
    }
}



/// Libsvm structure representing training data.
#[repr(C)]
struct LibsvmProblem {
    /// Number of rows in the training data.
    l: i32,
    y: *const f64,
    /// Rows of the X matrix. Because row lenghts
    /// are not stored anywhere, and do not need
    /// to be equal, `libsvm` uses index = -1 as
    /// a sentinel value.
    svm_node: *const *const LibsvmNode,
}


/// Safe version of `LibsvmProblem`.
pub struct SvmProblem {
    nodes: Vec<Vec<LibsvmNode>>,
    node_ptrs: Vec<*const LibsvmNode>,
    y: Vec<f64>,
}


/// Conert a row of the X matrix to its Libsvm representation.
fn row_to_nodes<T: NonzeroIterable>(row: T) -> Vec<LibsvmNode> {
    let mut nodes = Vec::new();

    for (index, value) in row.iter_nonzero() {
        nodes.push(LibsvmNode::new(index as i32, value as f64));
    }

    // Sentinel value for end of row
    nodes.push(LibsvmNode::new(-1, 0.0));

    nodes
}


impl SvmProblem {
    /// Create a new `SvmProblem` from training data.
    pub fn new<'a, T>(X: &'a T, y: &Array) -> SvmProblem
        where T: IndexableMatrix,
              &'a T: RowIterable
    {

        let mut nodes = Vec::with_capacity(X.rows());

        for row in X.iter_rows() {
            let row_nodes = row_to_nodes(row);
            nodes.push(row_nodes)
        }

        let node_ptrs = nodes.iter()
                             .map(|x| x.as_ptr())
                             .collect::<Vec<_>>();

        SvmProblem {
            nodes: nodes,
            node_ptrs: node_ptrs,
            y: y.data()
                .iter()
                .map(|&x| x as f64)
                .collect::<Vec<_>>(),
        }
    }

    /// Returns the unsafe object that can be passed into `libsvm`.
    fn build_problem(&self) -> LibsvmProblem {
        LibsvmProblem {
            l: self.nodes.len() as i32,
            y: self.y.as_ptr(),
            svm_node: self.node_ptrs.as_ptr(),
        }
    }
}


/// `libsvm` representation of training parameters.
#[repr(C)]
struct LibsvmParameter {
    svm_type: SvmType,
    kernel_type: KernelType,
    degree: i32,
    gamma: f64,
    coef0: f64,
    cache_size: f64,
    eps: f64,
    C: f64,
    nr_weight: i32,
    weight_label: *const i32,
    weight: *const f64,
    nu: f64,
    p: f64,
    shrinking: i32,
    probability: i32,
}


/// Safe representation of `LibsvmParameter`.
#[derive(Clone, Debug)]
#[derive(RustcEncodable, RustcDecodable)]
pub struct SvmParameter {
    svm_type: SvmType,
    kernel_type: KernelType,
    pub degree: i32,
    pub gamma: f64,
    pub coef0: f64,
    pub cache_size: f64,
    eps: f64,
    pub C: f64,
    nr_weight: i32,
    weight_label: Vec<i32>,
    weight: Vec<f64>,
    nu: f64,
    p: f64,
    shrinking: i32,
    probability: i32,
}


impl SvmParameter {
    pub fn new(svm_type: SvmType,
               kernel_type: KernelType,
               num_classes: usize,
               dim: usize)
               -> SvmParameter {

        SvmParameter {
            svm_type: svm_type,
            kernel_type: kernel_type,
            degree: 3,
            gamma: 1.0 / dim as f64,
            C: 1.0,
            coef0: 0.0,
            cache_size: 100.0,
            eps: 0.1,
            nr_weight: num_classes as i32,
            weight: vec![1.0; num_classes],
            weight_label: (0..num_classes).map(|x| x as i32).collect::<Vec<_>>(),
            nu: 0.5,
            p: 0.1,
            shrinking: 1,
            probability: 0,
        }
    }

    /// Returns the parameter object to be passed into
    /// `libsvm` functions.
    fn build_libsvm_parameter(&self) -> LibsvmParameter {
        LibsvmParameter {
            svm_type: self.svm_type.clone(),
            kernel_type: self.kernel_type.clone(),
            degree: self.degree,
            gamma: self.gamma,
            C: self.C,
            coef0: self.coef0,
            cache_size: self.cache_size,
            eps: self.eps,
            nr_weight: self.nr_weight,
            weight: self.weight.as_ptr(),
            weight_label: self.weight_label.as_ptr(),
            nu: self.nu,
            p: self.p,
            shrinking: self.shrinking,
            probability: self.probability,
        }
    }
}


/// The model object returned from and accepted by
/// `libsvm` functions.
#[repr(C)]
struct LibsvmModel {
    svm_parameter: LibsvmParameter,
    nr_class: i32,
    l: i32,
    SV: *const *const LibsvmNode,
    sv_coef: *const *const f64,
    rho: *const f64,
    probA: *const f64,
    probB: *const f64,
    sv_indices: *const i32,
    label: *const i32,
    nSV: *const i32,
    free_sv: i32,
}


/// Safe representation of `LibsvmModel`.
#[derive(Clone, Debug)]
#[derive(RustcEncodable, RustcDecodable)]
pub struct SvmModel {
    svm_parameter: SvmParameter,
    nr_class: i32,
    l: i32,
    SV: Vec<Vec<LibsvmNode>>,
    sv_coef: Vec<Vec<f64>>,
    rho: Vec<f64>,
    probA: Vec<f64>,
    probB: Vec<f64>,
    sv_indices: Vec<i32>,
    label: Vec<i32>,
    nSV: Vec<i32>,
    free_sv: i32,
}


impl SvmModel {
    fn new(param: SvmParameter, model_ptr: *const LibsvmModel) -> SvmModel {
        unsafe {
            SvmModel {
                svm_parameter: param,
                nr_class: ((*model_ptr)).nr_class,
                l: ((*model_ptr)).l,
                SV: SvmModel::get_SV(model_ptr),
                sv_coef: SvmModel::get_sv_coef(model_ptr),
                rho: SvmModel::get_rho(model_ptr),
                probA: vec![0.0],
                probB: vec![0.0],
                sv_indices: vec![0],
                label: SvmModel::get_label(model_ptr),
                nSV: SvmModel::get_nSV(model_ptr),
                free_sv: 0,
            }
        }
    }

    fn get_libsvm_model(&self,
                        SV_ptrs: &mut Vec<*const LibsvmNode>,
                        sv_coef_ptrs: &mut Vec<*const f64>)
                        -> LibsvmModel {

        SV_ptrs.clear();
        sv_coef_ptrs.clear();

        for x in self.SV.iter() {
            SV_ptrs.push(x.as_ptr());
        }

        for x in self.sv_coef.iter() {
            sv_coef_ptrs.push(x.as_ptr());
        }

        LibsvmModel {
            svm_parameter: self.svm_parameter.build_libsvm_parameter(),
            nr_class: self.nr_class,
            l: self.l,
            SV: SV_ptrs.as_ptr(),
            sv_coef: sv_coef_ptrs.as_ptr(),
            rho: self.rho.as_ptr(),
            probA: self.probA.as_ptr(),
            probB: self.probB.as_ptr(),
            sv_indices: self.sv_indices.as_ptr(),
            label: self.label.as_ptr(),
            nSV: self.nSV.as_ptr(),
            free_sv: self.free_sv,
        }
    }

    unsafe fn get_nSV(model_ptr: *const LibsvmModel) -> Vec<i32> {
        let nr_class = ((*model_ptr)).nr_class as usize;
        slice::from_raw_parts((*model_ptr).nSV, nr_class).to_owned()
    }

    unsafe fn get_label(model_ptr: *const LibsvmModel) -> Vec<i32> {
        let nr_class = (*model_ptr).nr_class as usize;
        slice::from_raw_parts((*model_ptr).label, nr_class).to_owned()
    }

    unsafe fn get_SV(model_ptr: *const LibsvmModel) -> Vec<Vec<LibsvmNode>> {
        let l = (*model_ptr).l;

        let mut sv_rows = Vec::with_capacity(l as usize);
        let sv_ptr = (*model_ptr).SV;

        for row in 0..l {

            let mut sv_row = Vec::new();
            let sv_row_ptr = *sv_ptr.offset(row as isize);
            let mut i = 0;

            loop {
                let node = (*sv_row_ptr.offset(i as isize)).clone();
                sv_row.push(node.clone());

                if node.index == -1 {
                    break;
                }

                i += 1;
            }

            sv_rows.push(sv_row);
        }

        sv_rows
    }

    unsafe fn get_rho(model_ptr: *const LibsvmModel) -> Vec<f64> {
        let mut nr_class = (*model_ptr).nr_class as usize;
        nr_class = nr_class * (nr_class - 1) / 2;
        slice::from_raw_parts((*model_ptr).rho, nr_class).to_owned()
    }

    unsafe fn get_sv_coef(model_ptr: *const LibsvmModel) -> Vec<Vec<f64>> {

        let nr_class = (*model_ptr).nr_class as usize;
        let l = (*model_ptr).l as usize;

        slice::from_raw_parts((*model_ptr).sv_coef, nr_class - 1)
            .iter()
            .map(|&x| slice::from_raw_parts(x, l).to_owned())
            .collect::<Vec<_>>()
    }
}


extern "C" {
    fn svm_train(prob: *const LibsvmProblem, param: *const LibsvmParameter) -> *const LibsvmModel;
    fn svm_predict_values(svm_model: *mut LibsvmModel,
                          svm_nodes: *const LibsvmNode,
                          out: *const f64)
                          -> f64;
    fn svm_free_and_destroy_model(svm_model: *const *const LibsvmModel);
    fn svm_check_parameter(problem: *const LibsvmProblem,
                           param: *const LibsvmParameter)
                           -> *const i8;
}


fn check(problem: *const LibsvmProblem, param: *const LibsvmParameter) -> Result<(), String> {
    unsafe {
        let message = svm_check_parameter(problem, param);

        match message.is_null() {
            true => Ok(()),
            false => Err(CStr::from_ptr(message).to_str().unwrap().to_owned()),
        }
    }
}


/// Fit a `libsvm` model.
pub fn fit<'a, T>(X: &'a T, y: &Array, parameters: &SvmParameter) -> Result<SvmModel, &'static str>
    where T: IndexableMatrix,
          &'a T: RowIterable
{

    let problem = SvmProblem::new(X, y);

    let libsvm_problem = problem.build_problem();
    let libsvm_param = parameters.build_libsvm_parameter();

    let model_ptr = unsafe {
        match check(&libsvm_problem as *const LibsvmProblem,
                    &libsvm_param as *const LibsvmParameter) {
            Ok(_) => {}
            Err(error_str) => {
                // A bit of a horrible out-of-band error reporting,
                // we should switch the model traits to String errors
                println!("Libsvm check error: {}", error_str);
                return Err("Invalid libsvm parameters.");
            }
        };
        svm_train(&libsvm_problem as *const LibsvmProblem,
                  &libsvm_param as *const LibsvmParameter)
    };

    let model = SvmModel::new(parameters.clone(), model_ptr);

    unsafe {
        // Free the model data allocated by libsvm,
        // we've got our own, sane copy.
        svm_free_and_destroy_model(&model_ptr);
    }

    Ok(model)
}


/// Call `libsvm` to get predictions (both predicted classes
/// and OvO decision function values.
pub fn predict<'a, T>(model: &SvmModel, X: &'a T) -> (Array, Array)
    where T: IndexableMatrix,
          &'a T: RowIterable
{

    let x_rows = X.rows();

    let num_classes = model.nr_class as usize;
    let ovo_num_classes = num_classes * (num_classes - 1) / 2;

    // We are actually mutating this in C, but convincing rustc that is
    // safe is a bit of a pain
    let df = vec![0.0; x_rows * ovo_num_classes];
    let mut df_slice = &df[..];

    let mut predicted_class = Vec::with_capacity(x_rows);

    // Allocate space for pointers to support vector components,
    // we don't need them after we're finished here
    // so they will be freed.
    let mut sv_ptrs = Vec::new();
    let mut sv_coef_ptrs = Vec::new();

    let mut libsvm_model = model.get_libsvm_model(&mut sv_ptrs, &mut sv_coef_ptrs);

    for (_, row) in X.iter_rows().enumerate() {
        let nodes = row_to_nodes(row);
        unsafe {
            predicted_class.push(svm_predict_values(&mut libsvm_model as * mut LibsvmModel,
                                                        nodes.as_ptr(),
                                                        df_slice.as_ptr())
                                     as f32);
        }
        df_slice = &df_slice[ovo_num_classes..];
    }

    let df_data = df.iter().map(|&x| x as f32).collect::<Vec<_>>();
    let mut df_array = Array::from(df_data);
    df_array.reshape(x_rows, ovo_num_classes);

    (df_array, Array::from(predicted_class))
}
