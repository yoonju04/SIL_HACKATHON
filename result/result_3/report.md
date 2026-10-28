# 파이프라인 실행 보고서 (X-5 full_code_aware, 영상 해설 on)

- 데이터: **실측 (CARLA + Token Factory)**
- 출력 폴더: `results\real_gen_1028`
- 런 13개 (유효 13, 전환 8), final_safe_recovery 4개, 실패 5개
- 경계 양쪽 관측: 예; |lane_recovery_robustness| < 0.2 런 0개
- 첫 실패까지 런 수: 3, 고유 실패 조건: 4
- GP 경계 부호 정확도: {'n': 8, 'accuracy': 1.0, 'truth': 'leave-one-out on observed runs'}
- 영상 해설: 5건, 검증 통과 0건 (환각률 1.0)
- recovery_outcome 분포: {'safe_recovery': 5, 'no_switch': 5, 'unrecovered': 3}
- trigger_source 분포: {'solver_failure': 3, 'None': 5, 'tracking_divergence': 3, 'stale_sequence': 2}
- SR 검사 (engine/sr_checks.py, 참고) sr_outcome 분포: {'F2_switch_transient': 2, 'F1_no_or_late_switch': 7, 'pass': 2, 'F3_not_recovered': 1, 'invalid': 1}; 화면: `screen/campaign.html`

## 주의
- 판정 임계값(0.3 m, 0.1 rad, 1.0 s 유지, 3.0 s 창, 4.41 m/s², 5.0 m/s³, 충돌 -2.0; 09 safety_spec.yaml)은 PROVISIONAL 이다. 인증 기준이 아니다.
- 측정된 복구 성공 영역은 수학적 안전 보증이 아니다.
- 영상 해설은 판정을 바꾸지 않는다. 검증을 통과한 해설만 피드백에 쓴다.
- monitor_rule_candidates.yaml 은 사람 검토용 제안이다 (자동 적용 안 함).

## 캠페인 인사이트 (09 monitor explain.py, Cosmos 텍스트 분석)
- 상태: error, 인사이트 0개, 모델 호출 2회, 오류: Token Factory HTTP 404
- 결과: `insights/insights.json`, 화면: `insights/campaign.html` (SR 모니터 기준 판정, 참고용)
