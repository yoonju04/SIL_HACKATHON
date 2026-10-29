# 파이프라인 실행 보고서 (X-5 full_code_aware, 영상 해설 on)

- 데이터: **실측 (CARLA + Token Factory)**
- 출력 폴더: `results\real_int4_16_1029`
- 런 16개 (유효 16, 전환 11), final_safe_recovery 7개, 실패 4개
- 경계 양쪽 관측: 예; |safety_margin| < 0.2 런 3개 (GP 타깃 < 0: 4개)
- 첫 실패까지 런 수: 2, 고유 실패 조건: 3
- GP 경계 부호 정확도: {'n': 11, 'accuracy': 0.36363636363636365, 'truth': 'leave-one-out on observed runs', 'truth_safe_fraction': 0.6363636363636364, 'majority_baseline': 0.6363636363636364, 'balanced_accuracy': 0.39285714285714285, 'predicted_safe_fraction': 0.36363636363636365, 'vs_final_safe': {'n': 11, 'accuracy': 0.36363636363636365, 'truth_safe_fraction': 0.6363636363636364, 'majority_baseline': 0.6363636363636364, 'balanced_accuracy': 0.39285714285714285, 'predicted_safe_fraction': 0.36363636363636365}}
- GP 부호 vs final_safe_recovery (#9): {'n': 11, 'accuracy': 0.36363636363636365, 'truth_safe_fraction': 0.6363636363636364, 'majority_baseline': 0.6363636363636364, 'balanced_accuracy': 0.39285714285714285, 'predicted_safe_fraction': 0.36363636363636365}
- 영상 해설: 3건, 검증 통과 2건 (환각률 0.33333333333333337)
- recovery_outcome 분포: {'safe_recovery': 11, 'no_switch': 5}
- trigger_source 분포: {'steer_rate_exceeded': 7, 'None': 5, 'stale_sequence': 1, 'tracking_divergence': 3}
- SR 검사 (engine/sr_checks.py, 참고) sr_outcome 분포: {'pass': 13, 'F1_no_or_late_switch': 3}; 화면: `screen/campaign.html`

## 경계 지표 (integration3)
- GP 타깃 safety_margin: |값| < 0.2 3개, < 0.5 4개, < 0 (불안전) 4개 (타깃이 있는 런 11개); 제한 기준 (binding): {'lat_jerk': 11}; 참고 |lane_recovery_robustness| < 0.2: 2개 / 11개
- 같은 고장 안에서 안전·불안전이 둘 다 나온 고장: ['bias', 'saturate']
- Nemotron A 후보 비율 (실행된 런): 0.375; 출처: {'nemotron_a': 5, 'space_filling': 9, 'counterfactual': 1, 'nemotron_a_refine': 1}
  - bias: 1/4 final_safe, {'no_switch': 2, 'safe_recovery': 2}
  - drift: 0/2 final_safe, {'safe_recovery': 2}
  - intermittent_bias: 2/2 final_safe, {'safe_recovery': 2}
  - output_delay: 0/3 final_safe, {'no_switch': 3}
  - saturate: 3/4 final_safe, {'safe_recovery': 4}
  - stale: 1/1 final_safe, {'safe_recovery': 1}
- GP 예측 다양성 recovery_success: 런 4개 중 서로 다른 값 4개, 범위 [-13.778533, -1.5e-05]
- GP 예측 다양성 switched_indicator: 런 4개 중 서로 다른 값 4개, 범위 [0.442268, 0.965069]

## 주의
- 판정 임계값(0.3 m, 0.1 rad, 1.0 s 유지, 3.0 s 창, 4.41 m/s², 5.0 m/s³, 충돌 -2.0; 09 safety_spec.yaml)은 PROVISIONAL 이다. 인증 기준이 아니다.
- 측정된 복구 성공 영역은 수학적 안전 보증이 아니다.
- 영상 해설은 판정을 바꾸지 않는다. 검증을 통과한 해설만 피드백에 쓴다.
- monitor_rule_candidates.yaml 은 사람 검토용 제안이다 (자동 적용 안 함).

## 캠페인 인사이트 (09 monitor explain.py, Cosmos 텍스트 분석)
- 상태: complete, 인사이트 5개, 모델 호출 7회
- 결과: `insights/insights.json`, 화면: `insights/campaign.html` (SR 모니터 기준 판정, 참고용)
