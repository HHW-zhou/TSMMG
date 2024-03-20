#!/bin/sh

python eval.py --sample_size=765435 --epochs=150 --eval_type=drd2 --cuda=0 &&
python eval.py --sample_size=765435 --epochs=150 --eval_type=drd2_qed --cuda=0  &&
python eval.py --sample_size=765435 --epochs=150 --eval_type=drd2_qed_sa --cuda=0  &&
python eval.py --sample_size=765435 --epochs=150 --eval_type=gsk3 --cuda=0  &&
python eval.py --sample_size=765435 --epochs=150 --eval_type=gsk3_qed --cuda=0  &&
python eval.py --sample_size=765435 --epochs=150 --eval_type=gsk3_qed_sa --cuda=0  &&
python eval.py --sample_size=765435 --epochs=150 --eval_type=jnk3 --cuda=0  &&
python eval.py --sample_size=765435 --epochs=150 --eval_type=jnk3_qed --cuda=0  &&
python eval.py --sample_size=765435 --epochs=150 --eval_type=jnk3_qed_sa --cuda=0 ;