#!/bin/sh

python eval.py --sample_size=765435 --epochs=150 --eval_type=bbb --cuda=1 &&
python eval.py --sample_size=765435 --epochs=150 --eval_type=bbb_qed --cuda=1  &&
python eval.py --sample_size=765435 --epochs=150 --eval_type=bbb_qed_sa --cuda=1  &&
python eval.py --sample_size=765435 --epochs=150 --eval_type=hia --cuda=1  &&
python eval.py --sample_size=765435 --epochs=150 --eval_type=hia_qed --cuda=1  &&
python eval.py --sample_size=765435 --epochs=150 --eval_type=hia_qed_sa --cuda=1  &&
python eval.py --sample_size=765435 --epochs=150 --eval_type=pgps --cuda=1  &&
python eval.py --sample_size=765435 --epochs=150 --eval_type=pgps_qed --cuda=1  &&
python eval.py --sample_size=765435 --epochs=150 --eval_type=pgps_qed_sa --cuda=1 ;