PROJDIR=`pwd`
run_file=${PROJDIR}/run_get_best_param.sh
rm $PROJDIR/o.get_best_param $PROJDIR/e.get_best_param  $run_file
echo "/users/mtaranov/local/anaconda2/bin/python $PROJDIR/get_best_param_rf.py" >> $run_file
echo "/users/mtaranov/local/anaconda2/bin/python $PROJDIR/get_best_param_svm_rbf.py" >> $run_file
chmod 777 $run_file
qsub -o $PROJDIR/o.get_best_param -e $PROJDIR/e.get_best_param $run_file
