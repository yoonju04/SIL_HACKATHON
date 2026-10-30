# 파이프라인 실행 보고서 (X-5 full_code_aware, 영상 해설 on)

- 데이터: **실측 (CARLA + Token Factory)**
- 출력 폴더: `results\real_int4_gp_1030`
- 런 32개 (유효 30, 전환 22), final_safe_recovery 14개, 실패 8개
- 경계 양쪽 관측: 예; |safety_margin| < 0.2 런 11개 (GP 타깃 < 0: 8개)
- 첫 실패까지 런 수: 4, 고유 실패 조건: 4
- GP 경계 부호 정확도: {'n': 22, 'accuracy': 0.7272727272727273, 'truth': 'leave-one-out on observed runs', 'truth_safe_fraction': 0.6363636363636364, 'majority_baseline': 0.6363636363636364, 'balanced_accuracy': 0.7589285714285714, 'predicted_safe_fraction': 0.45454545454545453, 'vs_final_safe': {'n': 22, 'accuracy': 0.7272727272727273, 'truth_safe_fraction': 0.6363636363636364, 'majority_baseline': 0.6363636363636364, 'balanced_accuracy': 0.7589285714285714, 'predicted_safe_fraction': 0.45454545454545453}}
- GP 부호 vs final_safe_recovery (#9): {'n': 22, 'accuracy': 0.7272727272727273, 'truth_safe_fraction': 0.6363636363636364, 'majority_baseline': 0.6363636363636364, 'balanced_accuracy': 0.7589285714285714, 'predicted_safe_fraction': 0.45454545454545453}
- 영상 해설: 14건, 검증 통과 11건 (환각률 0.2142857142857143)
- recovery_outcome 분포: {'safe_recovery': 19, 'no_switch': 8, 'recovered_after_departure': 2, 'simulation_invalid': 2, 'unsettled': 1}
- trigger_source 분포: {'steer_rate_exceeded': 14, 'None': 8, 'tracking_divergence': 9, 'stale_sequence': 1}
- SR 검사 (engine/sr_checks.py, 참고) sr_outcome 분포: {'pass': 23, 'F1_no_or_late_switch': 7, 'invalid': 2}; 화면: `screen/campaign.html`

## 경계 지표 (integration3)
- GP 타깃 safety_margin: |값| < 0.2 11개, < 0.5 17개, < 0 (불안전) 8개 (타깃이 있는 런 30개); 제한 기준 (binding): {'lat_acc': 3, 'lane': 11, 'lat_jerk': 8, 'settle': 8}; 참고 |lane_recovery_robustness| < 0.2: 6개 / 22개
- 같은 고장 안에서 안전·불안전이 둘 다 나온 고장: ['saturate']
- Nemotron A 후보 비율 (실행된 런): 0.34375; 출처: {'space_filling': 11, 'nemotron_a': 11, 'perturbation': 7, 'counterfactual': 3}
  - bias: 0/10 final_safe, {'no_switch': 6, 'safe_recovery': 4}
  - drift: 0/3 final_safe, {'recovered_after_departure': 2, 'safe_recovery': 1}
  - intermittent_bias: 11/11 final_safe, {'safe_recovery': 11}
  - output_delay: 0/2 final_safe, {'no_switch': 2}
  - saturate: 2/3 final_safe, {'safe_recovery': 2, 'unsettled': 1}
  - stale: 1/1 final_safe, {'safe_recovery': 1}
- GP 예측 다양성 recovery_success: 런 24개 중 서로 다른 값 24개, 범위 [-0.565328, 10.244064]
- GP 예측 다양성 switched_indicator: 런 24개 중 서로 다른 값 24개, 범위 [0.126932, 0.980376]

## 주의
- 판정 임계값(0.3 m, 0.1 rad, 1.0 s 유지, 3.0 s 창, 4.41 m/s², 15.0 m/s³, 충돌 -2.0; 09 safety_spec.yaml)은 PROVISIONAL 이다. 인증 기준이 아니다.
- 측정된 복구 성공 영역은 수학적 안전 보증이 아니다.
- 영상 해설은 판정을 바꾸지 않는다. 검증을 통과한 해설만 피드백에 쓴다.
- monitor_rule_candidates.yaml 은 사람 검토용 제안이다 (자동 적용 안 함).

## 캠페인 인사이트 (09 monitor explain.py, Cosmos 텍스트 분석)
- 상태: complete, 인사이트 4개, 모델 호출 16회
- 결과: `insights/insights.json`, 화면: `insights/campaign.html` (SR 모니터 기준 판정, 참고용)

## 크리티컬 고장과 경계값 (코드 표 + Cosmos)
- 상태: error, 런 149개, 순위 drift, bias, saturate, intermittent_bias, stale, solver_fail, output_delay
- 결과: `insights/boundaries.md`, `insights/boundaries.json`
