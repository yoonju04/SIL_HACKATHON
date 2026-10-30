# 파이프라인 실행 보고서 (X-5 full_code_aware, 영상 해설 on)

- 데이터: **실측 (CARLA + Token Factory)**
- 출력 폴더: `results\real_int3_min2x_1029`
- 런 16개 (유효 15, 전환 9), final_safe_recovery 3개, 실패 6개
- 경계 양쪽 관측: 예; |safety_margin| < 0.2 런 2개 (GP 타깃 < 0: 6개)
- 첫 실패까지 런 수: 1, 고유 실패 조건: 6
- GP 경계 부호 정확도: {'n': 9, 'accuracy': 0.6666666666666666, 'truth': 'leave-one-out on observed runs', 'truth_safe_fraction': 0.3333333333333333, 'majority_baseline': 0.6666666666666666, 'balanced_accuracy': 0.6666666666666666, 'predicted_safe_fraction': 0.4444444444444444, 'vs_final_safe': {'n': 9, 'accuracy': 0.6666666666666666, 'truth_safe_fraction': 0.3333333333333333, 'majority_baseline': 0.6666666666666666, 'balanced_accuracy': 0.6666666666666666, 'predicted_safe_fraction': 0.4444444444444444}}
- GP 부호 vs final_safe_recovery (#9): {'n': 9, 'accuracy': 0.6666666666666666, 'truth_safe_fraction': 0.3333333333333333, 'majority_baseline': 0.6666666666666666, 'balanced_accuracy': 0.6666666666666666, 'predicted_safe_fraction': 0.4444444444444444}
- 영상 해설: 4건, 검증 통과 1건 (환각률 0.75)
- recovery_outcome 분포: {'safe_recovery': 9, 'no_switch': 6, 'simulation_invalid': 1}
- trigger_source 분포: {'steer_rate_exceeded': 5, 'tracking_divergence': 4, 'None': 6, 'solver_failure': 1}
- SR 검사 (engine/sr_checks.py, 참고) sr_outcome 분포: {'pass': 12, 'F1_no_or_late_switch': 3, 'invalid': 1}; 화면: `screen/campaign.html`

## 경계 지표 (integration3)
- GP 타깃 safety_margin: |값| < 0.2 2개, < 0.5 4개, < 0 (불안전) 6개 (타깃이 있는 런 9개); 제한 기준 (binding): {'lat_jerk': 9}; 참고 |lane_recovery_robustness| < 0.2: 1개 / 9개
- 같은 고장 안에서 안전·불안전이 둘 다 나온 고장: ['intermittent_bias', 'saturate']
- Nemotron A 후보 비율 (실행된 런): 0.375; 출처: {'nemotron_a': 5, 'space_filling': 8, 'nemotron_a_refine': 1, 'perturbation': 2}
  - bias: 0/6 final_safe, {'safe_recovery': 1, 'no_switch': 5}
  - drift: 0/2 final_safe, {'safe_recovery': 2}
  - intermittent_bias: 1/2 final_safe, {'safe_recovery': 2}
  - output_delay: 0/1 final_safe, {'no_switch': 1}
  - saturate: 2/3 final_safe, {'safe_recovery': 3}
  - solver_fail: 0/1 final_safe, {'safe_recovery': 1}
- GP 예측 다양성 recovery_success: 런 4개 중 서로 다른 값 4개, 범위 [-17.697048, -10.543861]
- GP 예측 다양성 switched_indicator: 런 4개 중 서로 다른 값 4개, 범위 [0.359395, 0.422105]

## 주의
- 판정 임계값(0.3 m, 0.1 rad, 1.0 s 유지, 3.0 s 창, 4.41 m/s², 5.0 m/s³, 충돌 -2.0; 09 safety_spec.yaml)은 PROVISIONAL 이다. 인증 기준이 아니다.
- 측정된 복구 성공 영역은 수학적 안전 보증이 아니다.
- 영상 해설은 판정을 바꾸지 않는다. 검증을 통과한 해설만 피드백에 쓴다.
- monitor_rule_candidates.yaml 은 사람 검토용 제안이다 (자동 적용 안 함).

## 캠페인 인사이트 (09 monitor explain.py, Cosmos 텍스트 분석)
- 상태: complete, 인사이트 3개, 모델 호출 8회
- 결과: `insights/insights.json`, 화면: `insights/campaign.html` (SR 모니터 기준 판정, 참고용)
