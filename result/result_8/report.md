# 파이프라인 실행 보고서 (X-5 full_code_aware, 영상 해설 on)

- 데이터: **실측 (CARLA + Token Factory)**
- 출력 폴더: `results\real_int3_overnight_1029`
- 런 84개 (유효 82, 전환 50), final_safe_recovery 25개, 실패 25개
- 경계 양쪽 관측: 예; |lane_recovery_robustness| < 0.2 런 6개
- 첫 실패까지 런 수: 5, 고유 실패 조건: 16
- GP 경계 부호 정확도: {'n': 50, 'accuracy': 1.0, 'truth': 'leave-one-out on observed runs', 'truth_safe_fraction': 1.0, 'majority_baseline': 1.0, 'balanced_accuracy': None, 'predicted_safe_fraction': 1.0}
- 영상 해설: 24건, 검증 통과 1건 (환각률 0.9583333333333334)
- recovery_outcome 분포: {'safe_recovery': 48, 'no_switch': 32, 'simulation_invalid': 2, 'unsettled': 2}
- trigger_source 분포: {'steer_rate_exceeded': 33, 'solver_failure': 5, 'None': 32, 'tracking_divergence': 13, 'stale_sequence': 1}
- SR 검사 (engine/sr_checks.py, 참고) sr_outcome 분포: {'pass': 69, 'F1_no_or_late_switch': 13, 'invalid': 2}; 화면: `screen/campaign.html`

## 경계 지표 (integration3)
- |lane_recovery_robustness| < 0.2: 6개, < 0.5: 11개 (복구 지표가 있는 런 50개)
- 같은 고장 안에서 안전·불안전이 둘 다 나온 고장: ['bias', 'intermittent_bias', 'saturate', 'solver_fail']
- Nemotron A 후보 비율 (실행된 런): 0.36904761904761907; 출처: {'space_filling': 31, 'nemotron_a_refine': 4, 'nemotron_a': 27, 'perturbation': 22}
  - bias: 5/27 final_safe, {'safe_recovery': 10, 'unsettled': 2, 'no_switch': 15}
  - drift: 0/5 final_safe, {'safe_recovery': 5}
  - intermittent_bias: 10/21 final_safe, {'safe_recovery': 15, 'no_switch': 6}
  - output_delay: 0/11 final_safe, {'no_switch': 11}
  - saturate: 5/12 final_safe, {'safe_recovery': 12}
  - solver_fail: 4/5 final_safe, {'safe_recovery': 5}
  - stale: 1/1 final_safe, {'safe_recovery': 1}
- GP 예측 다양성 recovery_success: 런 72개 중 서로 다른 값 72개, 범위 [8e-06, 9.747856]
- GP 예측 다양성 switched_indicator: 런 72개 중 서로 다른 값 69개, 범위 [0.14915, 0.952473]

## 주의
- 판정 임계값(0.3 m, 0.1 rad, 1.0 s 유지, 3.0 s 창, 4.41 m/s², 5.0 m/s³, 충돌 -2.0; 09 safety_spec.yaml)은 PROVISIONAL 이다. 인증 기준이 아니다.
- 측정된 복구 성공 영역은 수학적 안전 보증이 아니다.
- 영상 해설은 판정을 바꾸지 않는다. 검증을 통과한 해설만 피드백에 쓴다.
- monitor_rule_candidates.yaml 은 사람 검토용 제안이다 (자동 적용 안 함).

## 캠페인 인사이트 (09 monitor explain.py, Cosmos 텍스트 분석)
- 상태: complete, 인사이트 5개, 모델 호출 29회
- 결과: `insights/insights.json`, 화면: `insights/campaign.html` (SR 모니터 기준 판정, 참고용)
