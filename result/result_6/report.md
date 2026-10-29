# 파이프라인 실행 보고서 (X-5 full_code_aware, 영상 해설 on)

- 데이터: **실측 (CARLA + Token Factory)**
- 출력 폴더: `results\real_int3_1028`
- 런 8개 (유효 8, 전환 4), final_safe_recovery 1개, 실패 5개
- 경계 양쪽 관측: 예; |lane_recovery_robustness| < 0.2 런 0개
- 첫 실패까지 런 수: 1, 고유 실패 조건: 5
- GP 경계 부호 정확도: {'n': 4, 'accuracy': None, 'note': 'too few switched runs'}
- 영상 해설: 5건, 검증 통과 0건 (환각률 1.0)
- recovery_outcome 분포: {'unrecovered': 2, 'recovered_after_departure': 1, 'safe_recovery': 1, 'no_switch': 4}
- trigger_source 분포: {'tracking_divergence': 3, 'stale_sequence': 1, 'None': 4}
- SR 검사 (engine/sr_checks.py, 참고) sr_outcome 분포: {'F1_no_or_late_switch': 5, 'pass': 3}; 화면: `screen/campaign.html`

## 경계 지표 (integration3)
- |lane_recovery_robustness| < 0.2: 0개, < 0.5: 1개 (복구 지표가 있는 런 4개)
- 같은 고장 안에서 안전·불안전이 둘 다 나온 고장: 없음 (결과가 고장 종류만으로 갈림)
- Nemotron A 후보 비율 (실행된 런): 0.125; 출처: {'nemotron_a': 1, 'space_filling': 7}
  - bias: 0/3 final_safe, {'recovered_after_departure': 1, 'no_switch': 2}
  - intermittent_bias: 0/1 final_safe, {'no_switch': 1}
  - output_delay: 0/1 final_safe, {'no_switch': 1}
  - saturate: 0/2 final_safe, {'unrecovered': 2}
  - stale: 1/1 final_safe, {'safe_recovery': 1}
- GP 예측 다양성 recovery_success: 런 4개 중 서로 다른 값 4개, 범위 [-18.913107, 0.040383]
- GP 예측 다양성 switched_indicator: 런 4개 중 서로 다른 값 3개, 범위 [0.479229, 0.517346]

## 주의
- 판정 임계값(0.3 m, 0.1 rad, 1.0 s 유지, 3.0 s 창, 4.41 m/s², 5.0 m/s³, 충돌 -2.0; 09 safety_spec.yaml)은 PROVISIONAL 이다. 인증 기준이 아니다.
- 측정된 복구 성공 영역은 수학적 안전 보증이 아니다.
- 영상 해설은 판정을 바꾸지 않는다. 검증을 통과한 해설만 피드백에 쓴다.
- monitor_rule_candidates.yaml 은 사람 검토용 제안이다 (자동 적용 안 함).

## 캠페인 인사이트 (09 monitor explain.py, Cosmos 텍스트 분석)
- 상태: complete, 인사이트 3개, 모델 호출 4회
- 결과: `insights/insights.json`, 화면: `insights/campaign.html` (SR 모니터 기준 판정, 참고용)
