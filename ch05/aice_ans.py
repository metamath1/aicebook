"""
AICE Associate 자격인증 범용 채점 모듈
- 문제집 하단에서 aice_ans.grade_answers(globals())로 호출
- aice_ans.py와 같은 폴더에 answer_config.json 파일이 있어야 함
"""

import json
import os
from typing import Any, Dict, Optional, Tuple


# 이 모듈 파일이 위치한 디렉토리 경로
# 구글 코랩에서 aice_ans.py와 answer_config.json을 같은 폴더에 업로드하면
# 자동으로 같은 폴더에서 설정 파일을 찾음
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_config(config_path: Optional[str] = None) -> Dict:
    """
    설정 파일 로드
    - config_path가 None이면 모듈과 같은 폴더에서 answer_config.json을 찾음
    - config_path가 지정되면 해당 경로에서 로드
    """
    if config_path is None:
        config_path = os.path.join(MODULE_DIR, "answer_config.json")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def check_exact_number(answer: Any, expected: Any, tolerance: float = 0) -> Tuple[bool, str]:
    """숫자 정확히 일치 여부 확인 (허용 오차 지원)"""
    try:
        answer_val = float(answer)
        expected_val = float(expected)

        if tolerance == 0:
            is_correct = answer_val == expected_val
        else:
            is_correct = abs(answer_val - expected_val) <= tolerance

        if is_correct:
            return True, "정답입니다!"
        else:
            return False, f"오답입니다. {answer} != {expected}"
    except (TypeError, ValueError) as e:
        return False, f"숫자 형식 오류: {e}"


def check_exact_string(answer: Any, expected: str, case_sensitive: bool = True) -> Tuple[bool, str]:
    """문자열 정확히 일치 여부 확인"""
    try:
        answer_str = str(answer)
        expected_str = str(expected)

        if case_sensitive:
            is_correct = answer_str == expected_str
        else:
            is_correct = answer_str.lower() == expected_str.lower()

        if is_correct:
            return True, "정답입니다!"
        else:
            return False, f"오답입니다. '{answer}' != '{expected}'"
    except Exception as e:
        return False, f"문자열 비교 오류: {e}"


def check_number_range(answer: Any, min_val: Optional[float] = None,
                       max_val: Optional[float] = None) -> Tuple[bool, str]:
    """숫자 범위 검증"""
    try:
        answer_val = float(answer)

        if min_val is not None and answer_val < min_val:
            return False, f"오답입니다. {answer_val}은(는) 최소값 {min_val}보다 작습니다."

        if max_val is not None and answer_val > max_val:
            return False, f"오답입니다. {answer_val}은(는) 최대값 {max_val}보다 큽니다."

        return True, f"정답입니다! (값: {answer_val})"
    except (TypeError, ValueError) as e:
        return False, f"숫자 형식 오류: {e}"


def check_history(history_dict: Any, requirements: Dict) -> Tuple[bool, str]:
    """딥러닝 학습 history 분석 및 검증

    requirements:
        - max_epochs: 최대 에폭 수
        - patience: EarlyStopping patience 값
        - monitor: 모니터링 메트릭 (예: 'val_accuracy', 'val_mae')
        - monitor_mode: 'max' 또는 'min' (accuracy는 max, loss/mae는 min)
    """
    try:
        if not isinstance(history_dict, dict):
            return False, "history.history가 딕셔너리가 아닙니다."

        max_epochs = requirements.get('max_epochs', 30)
        patience = requirements.get('patience', 3)
        monitor = requirements.get('monitor', 'val_accuracy')
        monitor_mode = requirements.get('monitor_mode', 'max')

        # 모니터링 메트릭 존재 확인
        if monitor not in history_dict:
            return False, f"모니터링 메트릭 '{monitor}'가 history에 없습니다."

        monitor_values = history_dict[monitor]
        actual_epochs = len(monitor_values)

        # 모든 에폭을 소진한 경우: patience 검증 생략
        if actual_epochs >= max_epochs:
            return True, f"정답입니다! (전체 {actual_epochs} 에폭 학습 완료)"

        # EarlyStopping이 작동한 경우: patience 검증
        if monitor_mode == 'max':
            best_value = max(monitor_values)
            best_epoch = monitor_values.index(best_value)
        else:  # min
            best_value = min(monitor_values)
            best_epoch = monitor_values.index(best_value)

        expected_stop_epoch = best_epoch + patience + 1  # 0-indexed + patience + 1

        # patience 검증: best_epoch 이후 patience 에폭 동안 개선이 없었는지 확인
        # 또는 best_value와 동일한 값이 이후에 있는 경우도 허용
        if actual_epochs == expected_stop_epoch:
            return True, f"정답입니다! (EarlyStopping 정상 작동, best epoch: {best_epoch + 1})"

        # 동점인 경우 체크
        if monitor_mode == 'max':
            best_indices = [i for i, v in enumerate(monitor_values) if v == best_value]
        else:
            best_indices = [i for i, v in enumerate(monitor_values) if v == best_value]

        for idx in best_indices:
            if actual_epochs == idx + patience + 1:
                return True, f"정답입니다! (EarlyStopping 정상 작동, best epoch: {idx + 1})"

        return False, f"EarlyStopping patience 설정이 올바르지 않습니다. (실제 에폭: {actual_epochs}, 예상: {expected_stop_epoch})"

    except Exception as e:
        return False, f"history 검증 오류: {e}"


def check_figure(fig: Any, requirements: Dict) -> Tuple[bool, str]:
    """matplotlib figure 구조 검증

    requirements:
        - num_axes: 필요한 axes 개수
        - lines_per_ax: 각 ax당 필요한 라인 수 (리스트 또는 단일 값)
        - titles: 각 ax의 제목 리스트 (선택)
        - xlabels: 각 ax의 x축 레이블 리스트 (선택)
        - ylabels: 각 ax의 y축 레이블 리스트 (선택)
    """
    try:
        # matplotlib.figure.Figure 타입 체크
        fig_type = type(fig).__name__
        if 'Figure' not in fig_type:
            return False, f"fig가 matplotlib Figure 객체가 아닙니다. (타입: {fig_type})"

        axes = fig.axes
        num_axes = requirements.get('num_axes', 2)

        # axes 개수 확인
        if len(axes) != num_axes:
            return False, f"axes 개수가 올바르지 않습니다. (실제: {len(axes)}, 필요: {num_axes})"

        # 각 ax의 라인 수 확인
        lines_per_ax = requirements.get('lines_per_ax', 2)
        if isinstance(lines_per_ax, int):
            lines_per_ax = [lines_per_ax] * num_axes

        for i, ax in enumerate(axes):
            lines = ax.get_lines()
            if len(lines) < lines_per_ax[i]:
                return False, f"ax[{i}]의 라인 수가 부족합니다. (실제: {len(lines)}, 필요: {lines_per_ax[i]})"

        # 제목 확인 (선택)
        titles = requirements.get('titles')
        if titles:
            for i, ax in enumerate(axes):
                actual_title = ax.get_title()
                if actual_title != titles[i]:
                    return False, f"ax[{i}]의 제목이 올바르지 않습니다. (실제: '{actual_title}', 필요: '{titles[i]}')"

        # x축 레이블 확인 (선택)
        xlabels = requirements.get('xlabels')
        if xlabels:
            for i, ax in enumerate(axes):
                actual_xlabel = ax.get_xlabel()
                if actual_xlabel != xlabels[i]:
                    return False, f"ax[{i}]의 x축 레이블이 올바르지 않습니다. (실제: '{actual_xlabel}', 필요: '{xlabels[i]}')"

        # y축 레이블 확인 (선택)
        ylabels = requirements.get('ylabels')
        if ylabels:
            for i, ax in enumerate(axes):
                actual_ylabel = ax.get_ylabel()
                if actual_ylabel != ylabels[i]:
                    return False, f"ax[{i}]의 y축 레이블이 올바르지 않습니다. (실제: '{actual_ylabel}', 필요: '{ylabels[i]}')"

        return True, "정답입니다!"

    except Exception as e:
        return False, f"figure 검증 오류: {e}"


def check_confusion_matrix(cm: Any, requirements: Dict) -> Tuple[bool, str]:
    """sklearn confusion_matrix 결과 검증

    requirements:
        - num_classes: 예상 클래스 개수 (선택)
    """
    try:
        # numpy ndarray 타입 체크
        type_name = type(cm).__name__
        if 'ndarray' not in type_name:
            return False, f"numpy ndarray가 아닙니다. (타입: {type_name})"

        # 2D 배열 확인
        if len(cm.shape) != 2:
            return False, f"2D 배열이 아닙니다. (shape: {cm.shape})"

        # 정방행렬 확인
        if cm.shape[0] != cm.shape[1]:
            return False, f"정방행렬이 아닙니다. (shape: {cm.shape})"

        # 클래스 개수 확인 (선택)
        num_classes = requirements.get('num_classes')
        if num_classes and cm.shape[0] != num_classes:
            return False, f"클래스 개수 불일치 (실제: {cm.shape[0]}, 필요: {num_classes})"

        return True, f"정답입니다! ({cm.shape[0]}x{cm.shape[1]} confusion matrix)"

    except Exception as e:
        return False, f"confusion_matrix 검증 오류: {e}"


def check_classification_report(report: Any, requirements: Dict) -> Tuple[bool, str]:
    """sklearn classification_report(output_dict=True) 결과 검증

    requirements:
        - num_classes: 예상 클래스 개수 (선택)
    """
    try:
        # dict 타입 확인
        if not isinstance(report, dict):
            return False, f"dict 타입이 아닙니다. (타입: {type(report).__name__})"

        # 필수 메타 키 존재 확인
        meta_keys = ['accuracy', 'macro avg', 'weighted avg']
        missing_keys = [k for k in meta_keys if k not in report]
        if missing_keys:
            return False, f"필수 키 누락: {missing_keys}"

        # accuracy 값이 숫자인지 확인
        if not isinstance(report['accuracy'], (int, float)):
            return False, f"'accuracy' 값이 숫자가 아닙니다. (타입: {type(report['accuracy']).__name__})"

        # macro avg, weighted avg가 dict인지 확인
        for key in ['macro avg', 'weighted avg']:
            if not isinstance(report[key], dict):
                return False, f"'{key}'가 딕셔너리가 아닙니다. (타입: {type(report[key]).__name__})"

        # 클래스 개수 확인 (선택)
        num_classes = requirements.get('num_classes')
        if num_classes:
            class_keys = [k for k in report.keys() if k not in meta_keys]
            if len(class_keys) != num_classes:
                return False, f"클래스 개수 불일치 (실제: {len(class_keys)}, 필요: {num_classes})"

        # 클래스 키 개수 계산
        class_keys = [k for k in report.keys() if k not in meta_keys]
        return True, f"정답입니다! ({len(class_keys)}개 클래스 classification report)"

    except Exception as e:
        return False, f"classification_report 검증 오류: {e}"


def check_keras_model(model: Any, requirements: Dict) -> Tuple[bool, str]:
    """Keras 모델 구조 검증

    requirements:
        - min_dense_layers: 최소 Dense 레이어 수 (출력층 제외)
        - min_dropout_layers: 최소 Dropout 레이어 수
        - dropout_rate: Dropout 비율 (허용 오차 0.01)
        - hidden_activation: 히든 레이어 활성화 함수 (예: 'relu', 'gelu')
        - output_activation: 출력 레이어 활성화 함수 (예: 'sigmoid', 'softmax', None)
    """
    try:
        # Keras 모델 타입 체크
        model_type = type(model).__name__
        model_module = type(model).__module__

        if 'keras' not in model_module.lower() and 'Sequential' not in model_type and 'Model' not in model_type:
            return False, f"Keras 모델이 아닙니다. (타입: {model_type})"

        # 모델 레이어 가져오기
        try:
            layers = model.layers
        except AttributeError:
            return False, "모델에서 레이어를 가져올 수 없습니다."

        # 레이어 분류
        dense_layers = []
        dropout_layers = []

        for layer in layers:
            layer_class = type(layer).__name__

            if 'Dense' in layer_class:
                dense_layers.append(layer)
            elif 'Dropout' in layer_class:
                dropout_layers.append(layer)

        # 출력층 분리 (마지막 Dense 레이어)
        if dense_layers:
            output_layer = dense_layers[-1]
            hidden_dense_layers = dense_layers[:-1]
        else:
            return False, "Dense 레이어가 없습니다."

        errors = []

        # 1. 최소 Dense 레이어 수 검사 (히든 레이어만)
        min_dense = requirements.get('min_dense_layers', 0)
        if min_dense > 0 and len(hidden_dense_layers) < min_dense:
            errors.append(f"히든 Dense 레이어 부족 (실제: {len(hidden_dense_layers)}, 필요: {min_dense}개 이상)")

        # 2. 최소 Dropout 레이어 수 검사
        min_dropout = requirements.get('min_dropout_layers', 0)
        if min_dropout > 0 and len(dropout_layers) < min_dropout:
            errors.append(f"Dropout 레이어 부족 (실제: {len(dropout_layers)}, 필요: {min_dropout}개 이상)")

        # 3. Dropout 비율 검사
        expected_dropout_rate = requirements.get('dropout_rate')
        if expected_dropout_rate is not None and dropout_layers:
            for i, dropout_layer in enumerate(dropout_layers):
                try:
                    actual_rate = float(dropout_layer.rate)
                    if abs(actual_rate - expected_dropout_rate) > 0.01:
                        errors.append(f"Dropout[{i}] 비율 불일치 (실제: {actual_rate}, 필요: {expected_dropout_rate})")
                except AttributeError:
                    pass  # rate 속성이 없으면 무시

        # 4. 히든 레이어 활성화 함수 검사
        expected_hidden_activation = requirements.get('hidden_activation')
        if expected_hidden_activation and hidden_dense_layers:
            for i, layer in enumerate(hidden_dense_layers):
                try:
                    # activation 함수 이름 가져오기
                    if hasattr(layer, 'activation'):
                        activation = layer.activation
                        if hasattr(activation, '__name__'):
                            actual_activation = activation.__name__
                        elif hasattr(activation, 'name'):
                            actual_activation = activation.name
                        else:
                            actual_activation = str(activation)

                        if actual_activation.lower() != expected_hidden_activation.lower():
                            errors.append(f"히든 Dense[{i}] 활성화 함수 불일치 (실제: {actual_activation}, 필요: {expected_hidden_activation})")
                except Exception:
                    pass

        # 5. 출력 레이어 활성화 함수 검사
        expected_output_activation = requirements.get('output_activation')
        if expected_output_activation:
            try:
                if hasattr(output_layer, 'activation'):
                    activation = output_layer.activation
                    if hasattr(activation, '__name__'):
                        actual_activation = activation.__name__
                    elif hasattr(activation, 'name'):
                        actual_activation = activation.name
                    else:
                        actual_activation = str(activation)

                    if actual_activation.lower() != expected_output_activation.lower():
                        errors.append(f"출력층 활성화 함수 불일치 (실제: {actual_activation}, 필요: {expected_output_activation})")
            except Exception:
                pass

        if errors:
            return False, "모델 구조 오류: " + "; ".join(errors)

        return True, f"정답입니다! (Dense: {len(hidden_dense_layers)}개 + 출력층, Dropout: {len(dropout_layers)}개)"

    except Exception as e:
        return False, f"모델 검증 오류: {e}"


def detect_answer_type(answer_value: Any) -> Optional[str]:
    """변수 타입을 감지하여 적절한 채점 유형 반환"""
    if answer_value is None:
        return None

    type_name = type(answer_value).__name__
    module_name = type(answer_value).__module__

    # Keras 모델 감지
    if 'keras' in module_name.lower() or 'tensorflow' in module_name.lower():
        if 'Sequential' in type_name or 'Model' in type_name or 'Functional' in type_name:
            return 'model_check'

    # matplotlib Figure 감지
    if 'Figure' in type_name and 'matplotlib' in module_name:
        return 'figure_check'

    # numpy ndarray 감지 (confusion_matrix)
    if 'ndarray' in type_name:
        return 'confusion_matrix_check'

    # dict 타입 감지 (classification_report 또는 history)
    if isinstance(answer_value, dict):
        # classification_report 감지 (더 특정적인 조건 먼저 체크)
        meta_keys = ['accuracy', 'macro avg', 'weighted avg']
        if all(key in answer_value for key in meta_keys):
            return 'classification_report_check'

        # history.history 감지 - val_로 시작하는 키가 있으면
        if any(key.startswith('val_') for key in answer_value.keys()):
            return 'history_check'

    # 숫자 타입
    if isinstance(answer_value, (int, float)):
        return 'exact_number'

    # 문자열 타입
    if isinstance(answer_value, str):
        return 'exact_string'

    return None


def grade_single_answer(answer_name: str, answer_value: Any, answer_config: Dict) -> Tuple[bool, str]:
    """단일 답안 채점

    type이 'auto'이면 변수 타입을 자동 감지하여 적절한 채점 함수 호출
    """
    answer_type = answer_config.get('type', 'auto')

    # auto 타입: 변수 타입을 감지하여 적절한 채점 유형 결정
    if answer_type == 'auto':
        detected_type = detect_answer_type(answer_value)
        if detected_type:
            answer_type = detected_type
        else:
            return False, f"변수 타입을 자동 감지할 수 없습니다. (타입: {type(answer_value).__name__})"

    if answer_type == 'exact_number':
        expected = answer_config.get('expected')
        tolerance = answer_config.get('tolerance', 0)
        return check_exact_number(answer_value, expected, tolerance)

    elif answer_type == 'exact_string':
        expected = answer_config.get('expected')
        case_sensitive = answer_config.get('case_sensitive', True)
        return check_exact_string(answer_value, expected, case_sensitive)

    elif answer_type == 'number_range':
        min_val = answer_config.get('min')
        max_val = answer_config.get('max')
        return check_number_range(answer_value, min_val, max_val)

    elif answer_type == 'history_check':
        requirements = answer_config.get('requirements', {})
        return check_history(answer_value, requirements)

    elif answer_type == 'figure_check':
        requirements = answer_config.get('requirements', {})
        return check_figure(answer_value, requirements)

    elif answer_type == 'model_check':
        requirements = answer_config.get('requirements', {})
        return check_keras_model(answer_value, requirements)

    elif answer_type == 'confusion_matrix_check':
        requirements = answer_config.get('requirements', {})
        return check_confusion_matrix(answer_value, requirements)

    elif answer_type == 'classification_report_check':
        requirements = answer_config.get('requirements', {})
        return check_classification_report(answer_value, requirements)

    elif answer_type == 'no_grade':
        return True, "채점 대상이 아닙니다."

    else:
        return False, f"알 수 없는 채점 유형: {answer_type}"


def grade_answers(globals_dict: Dict, config_path: Optional[str] = None) -> Dict:
    """
    메인 채점 함수

    Args:
        globals_dict: globals() 딕셔너리 (노트북의 전역 변수들)
        config_path: 설정 파일 경로 (None이면 모듈과 같은 폴더에서 answer_config.json을 찾음)

    Returns:
        채점 결과 딕셔너리
    """
    try:
        config = load_config(config_path)
        print(f"📁 설정 파일 로드 완료: {config.get('title', 'AICE 채점')}")
        print("=" * 40)
    except FileNotFoundError as e:
        print(f"🚨 {e}")
        print(f"💡 aice_ans.py와 answer_config.json을 같은 폴더에 업로드했는지 확인하세요.")
        return {}

    answers_config = config.get('answers', {})
    results = {}
    correct_count = 0
    total_count = 0

    # 답안 이름 정렬 (답안01, 답안02, ... 순서로)
    def sort_key(name):
        # 답안01, 답안02, 답안08_1, 답안08_2 등을 올바르게 정렬
        import re
        match = re.match(r'답안(\d+)(?:_(\d+))?', name)
        if match:
            main_num = int(match.group(1))
            sub_num = int(match.group(2)) if match.group(2) else 0
            return (main_num, sub_num)
        return (999, 0)

    sorted_answer_names = sorted(answers_config.keys(), key=sort_key)

    for answer_name in sorted_answer_names:
        answer_config = answers_config[answer_name]

        # 글로벌 변수에서 답안 가져오기
        if answer_name not in globals_dict:
            print(f"⚠️  {answer_name}: 변수가 정의되지 않았습니다.")
            print("-" * 20)
            results[answer_name] = (False, "변수 미정의")
            total_count += 1
            continue

        answer_value = globals_dict[answer_name]

        # 채점
        is_correct, message = grade_single_answer(answer_name, answer_value, answer_config)
        results[answer_name] = (is_correct, message)

        # 결과 출력
        if answer_config.get('type') == 'no_grade':
            print(f"⏭️  {answer_name}: {message}")
        elif is_correct:
            print(f"✅ {answer_name}: {message}")
            correct_count += 1
        else:
            print(f"❌ {answer_name}: {message}")

        if answer_config.get('type') != 'no_grade':
            total_count += 1

        print("-" * 20)

    # 최종 결과
    print(f"\n📊 채점 결과: {correct_count}/{total_count} 정답")
    if total_count > 0:
        score = (correct_count / total_count) * 100
        print(f"📈 점수: {score:.1f}%")

    return results


# 테스트용 코드
if __name__ == "__main__":
    print("AICE 채점 모듈이 로드되었습니다.")
    print("사용법: aice_ans.grade_answers(globals())")
