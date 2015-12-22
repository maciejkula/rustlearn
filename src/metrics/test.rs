

#[cfg(test)]
#[allow(unused_imports)]

mod generated_tests {
    use metrics::ranking::*;
    use prelude::*;
    use super::*;


    #[test]
    fn roc_auc_test_0() {
        let y_true = Array::from(&vec![vec![1.0], vec![1.0], vec![0.0], vec![1.0], vec![0.0],
                                       vec![1.0], vec![0.0], vec![0.0], vec![1.0], vec![1.0],
                                       vec![1.0], vec![0.0]]);
        let y_hat = Array::from(&vec![vec![0.07954696030457338],
                                      vec![2.1039526613830346],
                                      vec![-0.1201952328812757],
                                      vec![0.28145785314481458],
                                      vec![-0.020528698670943726],
                                      vec![-0.6625965461217288],
                                      vec![0.51824957300747021],
                                      vec![1.7345342477086778],
                                      vec![-0.059994608245898924],
                                      vec![0.74399695009589939],
                                      vec![-0.83144275519695576],
                                      vec![-1.6866803544338151]]);
        let expected = 0.542857142857;

        let computed = roc_auc_score(&y_true, &y_hat).unwrap();

        if !close(expected, computed) {
            println!("Expected: {} computed {}", expected, computed);
            assert!(false);
        }
    }



    #[test]
    fn roc_auc_test_1() {
        let y_true = Array::from(&vec![vec![1.0], vec![1.0], vec![0.0], vec![0.0], vec![0.0],
                                       vec![0.0], vec![1.0], vec![0.0], vec![1.0], vec![1.0],
                                       vec![0.0], vec![0.0], vec![0.0], vec![1.0], vec![0.0],
                                       vec![1.0], vec![0.0]]);
        let y_hat = Array::from(&vec![vec![2.3569813175920649],
                                      vec![-1.4131053859900191],
                                      vec![-0.69093190477591848],
                                      vec![0.68234881320614227],
                                      vec![1.4128307178741126],
                                      vec![-0.44858060509283809],
                                      vec![-0.16202171080189609],
                                      vec![-0.32167760715724231],
                                      vec![-0.4050358908439014],
                                      vec![-0.91355242609834042],
                                      vec![-0.58767766176923142],
                                      vec![-0.41762931916716467],
                                      vec![1.9276304159362743],
                                      vec![-0.59014460152415971],
                                      vec![1.7634996575897988],
                                      vec![-1.4095708314225499],
                                      vec![1.7177521313418966]]);
        let expected = 0.285714285714;

        let computed = roc_auc_score(&y_true, &y_hat).unwrap();

        if !close(expected, computed) {
            println!("Expected: {} computed {}", expected, computed);
            assert!(false);
        }
    }



    #[test]
    fn roc_auc_test_2() {
        let y_true = Array::from(&vec![vec![1.0], vec![1.0], vec![0.0], vec![0.0], vec![0.0],
                                       vec![0.0], vec![1.0], vec![0.0], vec![0.0], vec![0.0],
                                       vec![0.0], vec![0.0], vec![0.0], vec![0.0], vec![1.0],
                                       vec![1.0], vec![0.0], vec![1.0], vec![1.0]]);
        let y_hat = Array::from(&vec![vec![0.10083740530533056],
                                      vec![0.15940653274551736],
                                      vec![0.37693460700210579],
                                      vec![0.8922657731748358],
                                      vec![-1.41100542974677],
                                      vec![0.41256881260890538],
                                      vec![0.43368972171165837],
                                      vec![0.55035187001849606],
                                      vec![-1.7221833969755134],
                                      vec![0.81329178676026181],
                                      vec![0.42696502129269587],
                                      vec![-0.5548323268783486],
                                      vec![0.24526936366772165],
                                      vec![-0.36374239866159097],
                                      vec![-0.67775756424128875],
                                      vec![-0.13303985573211055],
                                      vec![-0.42190759446715842],
                                      vec![-0.3743798960725559],
                                      vec![-0.70021350348496858]]);
        let expected = 0.380952380952;

        let computed = roc_auc_score(&y_true, &y_hat).unwrap();

        if !close(expected, computed) {
            println!("Expected: {} computed {}", expected, computed);
            assert!(false);
        }
    }



    #[test]
    fn roc_auc_test_3() {
        let y_true = Array::from(&vec![vec![0.0], vec![0.0], vec![1.0], vec![1.0], vec![1.0],
                                       vec![1.0], vec![0.0], vec![0.0], vec![0.0], vec![0.0],
                                       vec![1.0], vec![0.0], vec![1.0], vec![1.0], vec![0.0],
                                       vec![0.0], vec![0.0], vec![1.0], vec![0.0]]);
        let y_hat = Array::from(&vec![vec![1.630120022802666],
                                      vec![-0.14536048719858083],
                                      vec![0.65049676373143017],
                                      vec![0.15284405171609525],
                                      vec![-1.4715936490387838],
                                      vec![0.27865358970468757],
                                      vec![1.6865979322330249],
                                      vec![0.35978955476998153],
                                      vec![-0.17471976535291622],
                                      vec![1.0996801927813424],
                                      vec![0.25032741077385451],
                                      vec![0.82490224349813057],
                                      vec![-0.50662363168830804],
                                      vec![-1.2946962928056853],
                                      vec![-0.045894795704719868],
                                      vec![0.53357624558762762],
                                      vec![-1.2006421658099216],
                                      vec![2.3646780111768262],
                                      vec![-0.67527614108293654]]);
        let expected = 0.397727272727;

        let computed = roc_auc_score(&y_true, &y_hat).unwrap();

        if !close(expected, computed) {
            println!("Expected: {} computed {}", expected, computed);
            assert!(false);
        }
    }



    #[test]
    fn roc_auc_test_4() {
        let y_true = Array::from(&vec![vec![0.0], vec![1.0], vec![0.0], vec![1.0], vec![0.0],
                                       vec![1.0], vec![0.0], vec![1.0]]);
        let y_hat = Array::from(&vec![vec![0.84546495595662241],
                                      vec![0.93718954838535729],
                                      vec![-0.51590516528591912],
                                      vec![1.0383661653664782],
                                      vec![-0.25439580710241955],
                                      vec![-0.83826262484583891],
                                      vec![0.49266173400470542],
                                      vec![0.38613665960028387]]);
        let expected = 0.625;

        let computed = roc_auc_score(&y_true, &y_hat).unwrap();

        if !close(expected, computed) {
            println!("Expected: {} computed {}", expected, computed);
            assert!(false);
        }
    }



}
