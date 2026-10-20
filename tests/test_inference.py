from pathlib import Path
import copy
import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from scripts.verify_artifacts import verify
from submission.inference import InferenceModel
from tests.synthetic_data import packet

ROOT = Path(__file__).resolve().parents[1]

class FrozenInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = packet()
        cls.day = str(int(cls.data['Barra_factor']['日期'].max()))
        cls.as_of = f'{cls.day[:4]}-{cls.day[4:6]}-{cls.day[6:]}'
        cls.model = InferenceModel(str(ROOT / 'submission/model'))
        cls.baseline = cls.model.predict(cls.as_of, cls.data)

    def check_equal(self, other):
        assert_frame_equal(self.baseline.sort_values('code').reset_index(drop=True),
                           other.sort_values('code').reset_index(drop=True),
                           check_exact=True)

    def test_artifact_identity_and_native_models(self):
        self.assertEqual(verify()['status'], 'pass')
        for spec, booster in self.model.models:
            self.assertEqual(booster.num_trees(), 350)
            self.assertEqual(booster.num_feature(), 229)
            self.assertEqual(booster.feature_name(), spec['features'])

    def test_output_contract(self):
        self.assertEqual(list(self.baseline), ['date', 'code', 'score'])
        expected = set(self.data['Barra_factor'].loc[
            self.data['Barra_factor']['日期'].eq(int(self.day)), '代码'])
        self.assertEqual(set(self.baseline.code), expected)
        self.assertTrue(self.baseline.code.is_unique)
        self.assertTrue(self.baseline.date.eq(self.as_of).all())
        self.assertTrue(np.isfinite(self.baseline.score).all())
        self.assertTrue(self.baseline.score.between(0, 1).all())

    def test_120_date_boundary_and_older_values(self):
        days = sorted(self.data['Barra_factor']['日期'].unique())
        recent = set(days[-120:])
        clipped = {k: v.loc[v['日期'].isin(recent)].copy() for k, v in self.data.items()}
        self.check_equal(self.model.predict(self.as_of, clipped))
        modified = copy.deepcopy(self.data)
        old = ~modified['MarketData']['日期'].isin(recent)
        modified['MarketData'].loc[old, ['open', 'high', 'low', 'close']] *= 3
        self.check_equal(self.model.predict(self.as_of, modified))

    def test_st_absent_or_arbitrary(self):
        absent = {k: v.drop(columns=['is_st'], errors='ignore') for k, v in self.data.items()}
        self.check_equal(self.model.predict(self.as_of, absent))
        changed = {k: v.assign(is_st='ignored synthetic field') for k, v in self.data.items()}
        self.check_equal(self.model.predict(self.as_of, changed))

    def test_future_rows_and_input_order(self):
        future = {}
        new_date = int((pd.Timestamp(self.as_of) + pd.Timedelta(days=1)).strftime('%Y%m%d'))
        for k, v in self.data.items():
            rows = v.loc[v['日期'].eq(int(self.day))].copy()
            rows['日期'] = new_date
            future[k] = pd.concat([v, rows], ignore_index=True).sample(frac=1, random_state=17)
        self.check_equal(self.model.predict(self.as_of, future))

    def test_input_and_parameters_unchanged_between_calls(self):
        before = {k: v.copy(deep=True) for k, v in self.data.items()}
        native = [b.model_to_string() for _, b in self.model.models]
        self.check_equal(self.model.predict(self.as_of, self.data))
        for name in before:
            assert_frame_equal(before[name], self.data[name], check_exact=True)
        self.assertEqual(native, [b.model_to_string() for _, b in self.model.models])

    def test_missing_signal_date_is_rejected(self):
        data = {k: v.loc[v['日期'].ne(int(self.day))].copy() for k, v in self.data.items()}
        with self.assertRaisesRegex(ValueError, 'Missing signal day'):
            self.model.predict(self.as_of, data)

    def test_conflicting_keys_are_rejected(self):
        data = copy.deepcopy(self.data)
        extra = data['MarketData'].iloc[[-1]].copy()
        extra['amount'] += 1
        data['MarketData'] = pd.concat([data['MarketData'], extra], ignore_index=True)
        with self.assertRaisesRegex(ValueError, 'conflicting duplicate'):
            self.model.predict(self.as_of, data)

if __name__ == '__main__':
    unittest.main()
