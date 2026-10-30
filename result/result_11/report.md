# 파이프라인 실행 보고서 (X-5 full_code_aware, 영상 해설 on)

- 데이터: **실측 (CARLA + Token Factory)**
- 출력 폴더: `results\real_int4_min_1030`
- 런 8개 (유효 8, 전환 5), final_safe_recovery 4개, 실패 1개
- 경계 양쪽 관측: 예; |safety_margin| < 0.2 런 1개 (GP 타깃 < 0: 1개)
- 첫 실패까지 런 수: 3, 고유 실패 조건: 1
- GP 경계 부호 정확도: {'n': 5, 'accuracy': None, 'note': 'too few switched runs'}
- GP 부호 vs final_safe_recovery (#9): None
- 영상 해설: 2건, 검증 통과 1건 (환각률 0.5)
- recovery_outcome 분포: {'no_switch': 3, 'safe_recovery': 4, 'recovered_after_departure': 1}
- trigger_source 분포: {'None': 3, 'steer_rate_exceeded': 2, 'tracking_divergence': 2, 'solver_failure': 1}
- SR 검사 (engine/sr_checks.py, 참고) sr_outcome 분포: {'pass': 6, 'F1_no_or_late_switch': 2}; 화면: `screen/campaign.html`

## 경계 지표 (integration3)
- GP 타깃 safety_margin: |값| < 0.2 1개, < 0.5 2개, < 0 (불안전) 1개 (타깃이 있는 런 8개); 제한 기준 (binding): {'lane': 4, 'lat_acc': 1, 'lat_jerk': 3}; 참고 |lane_recovery_robustness| < 0.2: 2개 / 5개
- 같은 고장 안에서 안전·불안전이 둘 다 나온 고장: 없음 (결과가 고장 종류만으로 갈림)
- Nemotron A 후보 비율 (실행된 런): 0.25; 출처: {'nemotron_a': 2, 'space_filling': 6}
  - bias: 0/2 final_safe, {'no_switch': 1, 'recovered_after_departure': 1}
  - drift: 1/1 final_safe, {'safe_recovery': 1}
  - intermittent_bias: 0/1 final_safe, {'no_switch': 1}
  - output_delay: 0/1 final_safe, {'no_switch': 1}
  - saturate: 2/2 final_safe, {'safe_recovery': 2}
  - solver_fail: 1/1 final_safe, {'safe_recovery': 1}

## 주의
- 판정 임계값(0.3 m, 0.1 rad, 1.0 s 유지, 3.0 s 창, 4.41 m/s², 15.0 m/s³, 충돌 -2.0; 09 safety_spec.yaml)은 PROVISIONAL 이다. 인증 기준이 아니다.
- 측정된 복구 성공 영역은 수학적 안전 보증이 아니다.
- 영상 해설은 판정을 바꾸지 않는다. 검증을 통과한 해설만 피드백에 쓴다.
- monitor_rule_candidates.yaml 은 사람 검토용 제안이다 (자동 적용 안 함).

## 캠페인 인사이트 (09 monitor explain.py, Cosmos 텍스트 분석)
- 상태: complete, 인사이트 3개, 모델 호출 4회
- 결과: `insights/insights.json`, 화면: `insights/campaign.html` (SR 모니터 기준 판정, 참고용)
