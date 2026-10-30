# 크리티컬 고장과 경계값

- 데이터: 런 149개 (이번 캠페인 30, 이전 결과 119); 판정은 지금 기준으로 다시 계산
- 위험 = 안전 여유 < 0, 감지 실패 = 늦은/못한 전환 (SR F1)

## 코드 계산 (결정론적)
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

## Cosmos 해석 (partial)

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
