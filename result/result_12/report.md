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

## 크리티컬 고장과 경계값

- 데이터: 런 149개 (이번 캠페인 30, 이전 결과 119); 판정은 지금 기준으로 다시 계산
- 위험 = 안전 여유 < 0, 감지 실패 = 늦은/못한 전환 (SR F1)

### 코드 계산 (결정론적)
1. drift: 13런, 위험 5 (38%), 늦은/못한 감지 13 (100%), 최악 여유 -0.955
   - rate_rad_s: 위험 경계 0.018~0.05 rad/s에서 결과가 갈림 (다른 조건 영향); 감지 경계 0.0166 rad/s부터 모두 위험, 그 아래는 미시험
2. bias: 47런, 위험 9 (19%), 늦은/못한 감지 11 (23%), 최악 여유 -0.616
   - magnitude_rad: 위험 경계 0.0427~0.1 rad에서 결과가 갈림 (다른 조건 영향); 감지 경계 0.0427~0.1 rad에서 결과가 갈림 (다른 조건 영향)
3. saturate: 24런, 위험 1 (4%), 늦은/못한 감지 0 (0%), 최악 여유 -0.1
   - duration_s: 위험 경계 0.35~1.0 s에서 결과가 갈림 (다른 조건 영향); 감지 경계 시험 범위 전체(0.1~1.0 s) 안전
   - magnitude: 위험 경계 0.6~1.0 fraction_of_max_steer에서 결과가 갈림 (다른 조건 영향); 감지 경계 시험 범위 전체(0.2~1.0 fraction_of_max_steer) 안전
   - ramp_s: 위험 경계 0.0~0.281 s에서 결과가 갈림 (다른 조건 영향); 감지 경계 0.281 s까지 모두 안전, 그 위는 미시험
4. intermittent_bias: 37런, 위험 0 (0%), 늦은/못한 감지 2 (5%), 최악 여유 0.0
   - magnitude_rad: 위험 경계 시험 범위 전체(0.05~0.15 rad) 안전; 감지 경계 0.0691~0.15 rad에서 결과가 갈림 (다른 조건 영향)
   - on_s: 위험 경계 0.9896 s까지 모두 안전, 그 위는 미시험; 감지 경계 0.7123~0.9725 s에서 결과가 갈림 (다른 조건 영향)
   - off_s: 위험 경계 0.9656 s까지 모두 안전, 그 위는 미시험; 감지 경계 0.8167~0.9656 s에서 결과가 갈림 (다른 조건 영향)
5. stale: 3런, 위험 0 (0%), 늦은/못한 감지 0 (0%), 최악 여유 0.1
6. solver_fail: 7런, 위험 0 (0%), 늦은/못한 감지 0 (0%), 최악 여유 0.275
7. output_delay: 18런, 위험 0 (0%), 늦은/못한 감지 0 (0%), 최악 여유 0.333
   - delay_ticks: 위험 경계 시험 범위 전체(1~6) 안전; 감지 경계 시험 범위 전체(1~6) 안전

### Cosmos 해석 (partial)

1. **drift**: 가장 높은 harmful_rate(0.385)와 late_rate(1.0)를 보유하며, 최악의 safety margin(-0.955)을 기록함. 모든 실험에서 tracking_divergence로만 감지됨.
2. **bias**: harmful_rate(0.191)과 late_rate(0.234)가 높으며, safety margin이 -0.616에 이르는 심각한 위험성을 보임. 대부분의 실험에서 steer_rate_exceeded 또는 tracking_divergence로 감지됨.
3. **saturate**: harmful_rate(0.042)는 낮지만, safety margin이 -0.1에 이르는 심각한 위험성을 보임. 대부분의 실험에서 steer_rate_exceeded로 감지됨.

- drift / rate_rad_s (harm, all conditions): 0.018~0.05 [overlap, high] rate_rad_s가 0.018 이상일 때 safety margin이 음수로 전환됨.
- bias / magnitude_rad (harm, all conditions): 0.0427~0.1 [overlap, high] magnitude_rad가 0.0427 이상일 때 safety margin이 음수로 전환됨.
- bias / magnitude_rad (detection, all conditions): 0.0427~0.1 [overlap, high] magnitude_rad가 0.0427 이상일 때 steer_rate_exceeded 또는 tracking_divergence로 감지됨.
- saturate / duration_s (harm, all conditions): 0.35~1.0 [overlap, high] duration_s가 0.35 이상일 때 safety margin이 음수로 전환됨.
- saturate / duration_s (detection, all conditions): 0.1399~1.0 [estimated, high] duration_s가 0.1399 이상일 때 steer_rate_exceeded로 감지됨.
- saturate / magnitude (harm, all conditions): 0.6~1.0 [overlap, high] magnitude가 0.6 이상일 때 safety margin이 음수로 전환됨.
- saturate / magnitude (detection, all conditions): 0.3~1.0 [estimated, high] magnitude가 0.3 이상일 때 steer_rate_exceeded로 감지됨.
- saturate / ramp_s (harm, all conditions): 0.0~0.281 [overlap, high] ramp_s가 0.0 이상일 때 safety margin이 음수로 전환됨.
- saturate / ramp_s (detection, all conditions): 0.0~0.281 [estimated, high] ramp_s가 0.0 이상일 때 steer_rate_exceeded로 감지됨.

### 조건에 따른 변화
- bias의 magnitude_rad은 speed 25-30 km/h에서 0.0479~0.075, speed >=35 km/h에서 0.0427~0.0748로 경계가 좁아짐.
- drift의 rate_rad_s은 speed <=20 km/h에서 0.0492~0.05, speed >=35 km/h에서 0.018~0.0477로 경계가 좁아짐.
- saturate의 duration_s은 speed 25-30 km/h에서 0.35~0.672, friction >=0.7에서 0.35~1.0로 경계가 좁아짐.
- saturate의 magnitude는 speed 25-30 km/h에서 0.6~1.0, friction >=0.7에서 0.6~0.88로 경계가 좁아짐.
- saturate의 ramp_s은 speed 25-30 km/h에서 0.0~0.1872, friction >=0.7에서 0.0~0.2318로 경계가 좁아짐.

### 경계를 좁힐 다음 실험
- bias magnitude_rad = [0.04, 0.045, 0.05] @ 35 km/h, curve_exit: 현재 경계가 0.0427~0.1로 넓어, 0.04~0.05 사이의 정밀한 경계를 확인해야 함.
- drift rate_rad_s = [0.015, 0.017, 0.019] @ 20 km/h, curve_apex: detection 경계가 0.0166에서 시작하므로, 이하의 값에서 감지 여부를 확인해야 함.
- saturate duration_s = [0.3, 0.32, 0.34] @ 25 km/h, curve_entry: harm 경계가 0.35에서 시작하므로, 이하의 값에서 safety margin을 확인해야 함.

### 한계
- intermittent_bias와 no_problem_faults는 실험에서 안전 위협이 없어 분석 대상에서 제외됨.
- saturate의 ramp_s는 0.0에서 시작하므로, 0.0 이하의 값은 실험되지 않음.
- 일부 조건(예: speed 10 km/h, friction 0.5)은 실험되지 않아 경계 추정에 제한이 있음.
