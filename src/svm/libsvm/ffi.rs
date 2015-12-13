//! Internals of the `libsvm` FFI.
//!
//! Objects whose names start with `Libsvm` are for the most part
//! things that we pass or get directly from `libsvm`, and are highly
//! fragile.
//!
//! Their safe, memory-owning counterparts start with `Svm`.


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
    NU_SVR
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
    value: f64
}


impl LibsvmNode {
    fn new(index: i32, value: f64) -> LibsvmNode {
        LibsvmNode { index: index, value: value }
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
    svm_node: *const *const LibsvmNode
}


/// Safe version of `LibsvmProblem`.
pub struct SvmProblem {
    nodes: Vec<Vec<LibsvmNode>>,
    node_ptrs: Vec<* const LibsvmNode>,
    y: Vec<f64>
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
    pub fn new<T: RowIterable>(X: T, y: &Array) -> SvmProblem {

        let mut nodes = Vec::new();

        for row in X.iter_rows() {
            let row_nodes = row_to_nodes(row);
            nodes.push(row_nodes)
        }

        let node_ptrs = nodes.iter().map(|x| x.first().unwrap() as *const LibsvmNode)
            .collect::<Vec<_>>();

        SvmProblem { nodes: nodes,
                     node_ptrs: node_ptrs,
                     y: y.data().iter()
                     .map(|&x| x as f64).collect::<Vec<_>>() }
    }

    /// Returns the unsafe object that can be passed into `libsvm`.
    fn build_problem(&self) -> LibsvmProblem {
        LibsvmProblem { l: self.nodes.len() as i32,
                        y: self.y.first().unwrap() as *const f64,
                        svm_node: self.node_ptrs.first().unwrap() as *const *const LibsvmNode }
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
    probability: i32
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
    probability: i32
}


impl SvmParameter {
    pub fn new(svm_type: SvmType, kernel_type: KernelType,
           num_classes: usize, dim: usize) -> SvmParameter {

        SvmParameter { svm_type: svm_type,
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
                       probability: 0 }
    }

    /// Returns the parameter object to be passed into
    /// `libsvm` functions.
    fn build_libsvm_parameter(&self) -> LibsvmParameter {
        LibsvmParameter { svm_type: self.svm_type.clone(),
                          kernel_type: self.kernel_type.clone(),
                          degree: self.degree,
                          gamma: self.gamma,
                          C: self.C,
                          coef0: self.coef0,
                          cache_size: self.cache_size,
                          eps: self.eps,
                          nr_weight: self.nr_weight,
                          weight: self.weight.first().unwrap() as *const f64,
                          weight_label: self.weight_label.first().unwrap() as *const i32,
                          nu: self.nu,
                          p: self.p,
                          shrinking: self.shrinking,
                          probability: self.probability }
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
    free_sv: i32
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
    free_sv: i32
}


impl SvmModel {
    fn new(param: SvmParameter, model_ptr: *const LibsvmModel) -> SvmModel {
        unsafe {
            SvmModel { svm_parameter: param,
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
                       free_sv: 0 }
        }
    }

    fn get_libsvm_model(&self, SV_ptrs: &mut Vec<* const LibsvmNode>,
                        sv_coef_ptrs: &mut Vec<* const f64>) -> LibsvmModel {

        SV_ptrs.clear();
        sv_coef_ptrs.clear();

        for x in self.SV.iter() {
            SV_ptrs.push(x.first().unwrap() as * const LibsvmNode);
        }

        for x in self.sv_coef.iter() {
            sv_coef_ptrs.push(x.first().unwrap() as * const f64);
        }
            
        LibsvmModel { svm_parameter: self.svm_parameter.build_libsvm_parameter(),
                      nr_class: self.nr_class,
                      l: self.l,
                      SV: SV_ptrs.first().unwrap() as * const * const LibsvmNode,
                      sv_coef: sv_coef_ptrs.first().unwrap() as * const * const f64,
                      rho: self.rho.first().unwrap() as * const f64,
                      probA: self.probA.first().unwrap() as * const f64,
                      probB: self.probB.first().unwrap() as * const f64,
                      sv_indices: self.sv_indices.first().unwrap() as * const i32,
                      label: self.label.first().unwrap() as * const i32,
                      nSV: self.nSV.first().unwrap() as * const i32,
                      free_sv: self.free_sv }
    }

    unsafe fn get_nSV(model_ptr: *const LibsvmModel) -> Vec<i32> {
        let nr_class = ((*model_ptr)).nr_class;
        let mut nSV = Vec::with_capacity(nr_class as usize);
        let nsv_ptr = (*model_ptr).nSV;

        for i in 0..nr_class {
            nSV.push(*nsv_ptr.offset(i as isize));
        }

        nSV
    }

    unsafe fn get_label(model_ptr: *const LibsvmModel) -> Vec<i32> {

        let nr_class = (*model_ptr).nr_class;
        let mut labels = Vec::with_capacity(nr_class as usize);
        let label_ptr = (*model_ptr).label;

        for i in 0..nr_class {
            labels.push(*label_ptr.offset(i as isize));
        }

        labels
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
                    break
                }
                
                i += 1;
            }

            sv_rows.push(sv_row);
        }

        sv_rows
    }

    unsafe fn get_rho(model_ptr: *const LibsvmModel) -> Vec<f64> {
        let mut nr_class = (*model_ptr).nr_class;
        nr_class = nr_class * (nr_class - 1) / 2;

        let mut rho = Vec::with_capacity(nr_class as usize);

        let rho_ptr = (*model_ptr).rho;

        for i in 0..nr_class {
            rho.push(*rho_ptr.offset(i as isize));
        }

        rho
    }

    unsafe fn get_sv_coef(model_ptr: *const LibsvmModel) -> Vec<Vec<f64>> {

        let nr_class = (*model_ptr).nr_class;
        let l = (*model_ptr).l;

        let mut sv_coef = Vec::with_capacity((nr_class - 1) as usize);

        for i in 0..(nr_class - 1) {

            let row_ptr = *(*model_ptr).sv_coef.offset(i as isize);
            let mut coef_row = Vec::with_capacity(l as usize);

            for j in 0..l {
                coef_row.push(*row_ptr.offset(j as isize));
            }

            sv_coef.push(coef_row);
        }

        sv_coef
    }
        
}


extern {
    fn svm_train(prob: *const LibsvmProblem,
                 param: *const LibsvmParameter) -> *const LibsvmModel;
    fn svm_predict_values(svm_model: *mut LibsvmModel, svm_nodes: *const LibsvmNode, out: *const f64) -> f64;
    fn svm_free_and_destroy_model(svm_model: * const * const LibsvmModel);
}


/// Fit a `libsvm` model.
pub fn fit<T: RowIterable>(X: T, y: &Array, parameters: &SvmParameter) -> SvmModel {

    let problem = SvmProblem::new(X, y);

    let model_ptr = unsafe {
        svm_train(&problem.build_problem() as * const LibsvmProblem,
                  &parameters.build_libsvm_parameter() as * const LibsvmParameter)
    };

    let model = SvmModel::new(parameters.clone(), model_ptr);

    unsafe {
        // Free the model data allocated by libsvm,
        // we've got our own, sane copy.
        svm_free_and_destroy_model(&model_ptr);
    }

    model
}


/// Call `libsvm` to get predictions (both predicted classes
/// and OvO decision function values.
pub fn predict<T: RowIterable>(model: &SvmModel, X: T, x_rows: usize) -> (Array, Array) {

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
                                                    nodes.first().unwrap() as *const LibsvmNode,
                                                    df_slice.first().unwrap() as *const f64)
                                 as f32);
        }
        df_slice = &df_slice[ovo_num_classes..];
    }

    let df_data = df.iter().map(|&x| x as f32).collect::<Vec<_>>();
    let mut df_array = Array::from(df_data);
    df_array.reshape(x_rows, ovo_num_classes);

    (df_array, Array::from(predicted_class))
}
