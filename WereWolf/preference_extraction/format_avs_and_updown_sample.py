import json
import os
import argparse
import random
from MARBO.wolf.preference_extraction.utils_english import get_system_prompt
from tqdm import tqdm
from pathlib import Path
import re

TRAIN_RATIO = 0.85

def extract_identity(text):
    match = re.search(r'(?:你的身份是：|Your identity is: )(.+?)(?:。|\.|\n)', text)
    if match:
        role = match.group(1).strip()
        role_map = {
            "Villager": "村民",
            "Werewolf": "狼人",
            "Seer": "预言家",
            "Witch": "女巫",
            "Guard": "守卫",
            "Hunter": "猎人"
        }
        return role_map.get(role, role)
    else:
        return None
    
def create_message(role, content):
    return {"role": role, "content": content}


def _extract_day_from_phase(phase):
    match = re.match(r"(\d+)_", str(phase))
    if not match:
        return None
    return int(match.group(1))


def _extract_user_prompt_text(prompt):
    if isinstance(prompt, list):
        for message in prompt:
            if isinstance(message, dict) and message.get("role") == "user":
                return str(message.get("content", ""))
        for message in prompt:
            if isinstance(message, dict):
                return str(message.get("content", ""))
        return ""
    return str(prompt or "")


def _has_meaningful_subjective_info(prompt_text):
    match = re.search(
        r"3\.\s*Subjective Information[：:]\s*(.*?)(?:\n\d+\.\s|\Z)",
        prompt_text,
        flags=re.S,
    )
    if not match:
        return True

    subjective_info = match.group(1).strip()
    if not subjective_info:
        return False

    return subjective_info.lower() not in {"none", "null", "n/a"}


def _state_recon_keep_probability(day):
    if day is None:
        return 0.6
    if day <= 1:
        return 0.4
    if day == 2:
        return 0.6
    if day == 3:
        return 0.8
    return 1.0


def _should_keep_state_recon_sample(prompt_text, phase):
    if "state_reconstruction" not in str(phase):
        return True
    if not _has_meaningful_subjective_info(prompt_text):
        return False
    keep_prob = _state_recon_keep_probability(_extract_day_from_phase(phase))
    return random.random() < keep_prob
 
def process_content_sample(game_type, avs_type, data):
    prompt_list = []
    completion_list = []
    label_list = []
    if "good" in avs_type:
        label = True
    else:
        label = False
    for file_root, content in data.items():
        for i, (player_id, phase_list) in enumerate(content.items()):
            if not os.path.exists(os.path.join(file_root, f"Player_{player_id}.jsonl")):
                print("not exist", os.path.join(file_root, f"Player_{player_id}.jsonl"))
                continue
            with open(os.path.join(file_root, f"Player_{player_id}.jsonl"), "r") as f:
                for line in f:
                    player_game_play_info = json.loads(line.strip())
                    phase = player_game_play_info["phase"]
                    if phase in phase_list:
                        system_prompt = get_system_prompt(game_type)
                        input_prompt = player_game_play_info["prompt"]
                        response = player_game_play_info["response"]
                        if input_prompt is None or response is None or len(response)==0 or len(input_prompt)==0:
                            continue
                        if '\n** 游戏说明\n你现在正在玩一种叫做“狼人杀”的游戏。' in input_prompt \
                           or 'The rules and instructions for the Werewolf game are as follows:' in input_prompt: # filter gpt input
                            continue
                        
                        prompt = [create_message("system", system_prompt), 
                                  create_message("user", input_prompt)]
                        completion = [create_message("assistant", response)]

                        if not _should_keep_state_recon_sample(input_prompt, phase):
                            continue

                        # adjust sampling strategy
                        if 'bad_vote' in avs_type:
                            if phase == "1_day_vote":
                                if random.random() < 0.5:
                                    continue
                            elif phase == "2_day_vote":
                                if random.random() < 0.2:
                                    continue
                            if any(x in phase for x in ['3_day_vote', '4_day_vote', '5_day_vote']) and ('pk' not in phase): # add twice
                                if random.random() > 0.1:
                                    prompt_list.append(prompt)
                                    completion_list.append(completion)
                                    label_list.append(label)
                            elif "pk" in phase:
                                if random.random() > 0.01:
                                    prompt_list.append(prompt)
                                    completion_list.append(completion)
                                    label_list.append(label)
                        elif 'good_vote' in avs_type:
                            if any(x in phase for x in ['1_day_vote', '2_day_vote']) and ('pk' not in phase): # downsample
                                if random.random() > 0.4:
                                    continue
                            elif any(x in phase for x in ['3_day_vote', '4_day_vote']) and ('pk' not in phase): # add twice
                                if random.random() > 0.01:
                                    prompt_list.append(prompt)
                                    completion_list.append(completion)
                                    label_list.append(label)
                            elif 'pk' in phase:
                                prompt_list.append(prompt)
                                completion_list.append(completion)
                                label_list.append(label)
                        elif 'bad_speech' in avs_type:
                            if any(x in phase for x in ['1_day_speech', '2_day_speech']) and ('pk' not in phase): # downsample
                                if random.random() > 0.9:
                                    continue
                            elif any(x in phase for x in ['4_day_speech', '5_day_speech']) and ('pk' not in phase): # add twice
                                if random.random() > 0.01:
                                    prompt_list.append(prompt)
                                    completion_list.append(completion)
                                    label_list.append(label)
                            elif 'pk' in phase:
                                prompt_list.append(prompt)
                                completion_list.append(completion)
                                label_list.append(label)
                        elif 'good_action' in avs_type:
                            if any(x in phase for x in ['1_night_skill_wolf']): # add twice
                                if random.random() > 0.5:
                                    continue
                            elif any(x in phase for x in ['2_night_skill_witch', '3_night_skill', "4_night_skill"]): # add twice
                                if random.random() > 0.01:
                                    prompt_list.append(prompt)
                                    completion_list.append(completion)
                                    label_list.append(label)
                                    prompt_list.append(prompt)
                                    completion_list.append(completion)
                                    label_list.append(label)
                        elif 'bad_action' in avs_type:
                            if random.random() < 0.999:
                                prompt_list.append(prompt)
                                completion_list.append(completion)
                                label_list.append(label)

                        prompt_list.append(prompt)
                        completion_list.append(completion)
                        label_list.append(label)
                        
    return prompt_list, completion_list, label_list


def process_adversarial_sample(samples):
    prompt_list = []
    completion_list = []
    label_list = []

    for sample in samples:
        prompt = sample.get("prompt")
        completion = sample.get("completion")
        label = sample.get("label")
        phase = sample.get("phase", "")

        if prompt is None or completion is None or label is None:
            continue
        if not isinstance(prompt, list) or len(prompt) == 0:
            continue

        prompt_text = _extract_user_prompt_text(prompt)
        if not _should_keep_state_recon_sample(prompt_text, phase):
            continue

        if isinstance(completion, dict):
            completion = [create_message("assistant", json.dumps(completion, ensure_ascii=False))]
        elif isinstance(completion, list):
            normalized_completion = []
            for message in completion:
                if not isinstance(message, dict):
                    continue
                message_role = message.get("role", "assistant")
                message_content = message.get("content", "")
                if isinstance(message_content, (dict, list)):
                    message_content = json.dumps(message_content, ensure_ascii=False)
                else:
                    message_content = str(message_content)
                normalized_completion.append(create_message(message_role, message_content))
            completion = normalized_completion
        else:
            completion = [create_message("assistant", str(completion))]

        if len(completion) == 0:
            continue

        prompt_list.append(prompt)
        completion_list.append(completion)
        label_list.append(label)

    return prompt_list, completion_list, label_list


def process_good_speech_sample(jsonl_files):
    prompt_list = []
    completion_list = []
    label_list = []
    for file_path in jsonl_files:
        print(f"Processing file: {file_path}")
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                content = json.loads(line.strip())
                if content["prompt"] is None or content["response"] is None or len(content["response"]) == 0 or len(content["prompt"]) == 0:
                    continue
                role = extract_identity(content["prompt"])
                phase = content["phase"]
                prompt = [create_message("system", content["system_prompt"]), 
                          create_message("user", content["prompt"])]
                completion = [create_message("assistant", content["response"])]

                # 采样策略
                if content["judge_by_claude"]["final_judge"] == "accept":
                    label = True
                    if role == "村民":
                        if random.random() > 0.4:
                            continue
                    elif role == "狼人":
                        if random.random() < 0.4:
                            continue
                    if "pk" in phase:
                        if random.random() > 0.05:
                            prompt_list.append(prompt)
                            completion_list.append(completion)
                            label_list.append(label)
                    elif phase == "1_day_speech":
                        if random.random() > 0.5:
                            continue
                    elif any(x in phase for x in ['3_day_speech', '4_day_speech', '5_day_speech']):
                        if random.random() > 0.2:
                            prompt_list.append(prompt)
                            completion_list.append(completion)
                            label_list.append(label)
                else:
                    label = False
                    if role == "预言家":
                        prompt_list.append(prompt)
                        completion_list.append(completion)
                        label_list.append(label)
                        prompt_list.append(prompt)
                        completion_list.append(completion)
                        label_list.append(label)
                    elif role == "狼人":
                        if random.random() < 0.6:
                            continue
                    elif role == "村民":
                        if random.random() < 0.4:
                            continue
                    if 'pk' in phase:
                        prompt_list.append(prompt)
                        completion_list.append(completion)
                        label_list.append(label)
                        prompt_list.append(prompt)
                        completion_list.append(completion)
                        label_list.append(label)
                    elif any(x in phase for x in ['1_day_speech', '2_day_speech']):
                        if random.random() > 0.7:
                            continue
                    elif any(x in phase for x in ['3_day_speech', '4_day_speech', '5_day_speech']):
                        if random.random() > 0.01:
                            prompt_list.append(prompt)
                            completion_list.append(completion)
                            label_list.append(label)
                
                prompt_list.append(prompt)
                completion_list.append(completion)
                label_list.append(label)
    return prompt_list, completion_list, label_list

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--selected_path', type=str, default="./tmp")
    argparser.add_argument('--conflict_speech_path', type=str, default="./tmp")
    argparser.add_argument('--save_prefix', type=str, default="")
    argparser.add_argument('--game_type', type=str, default="9p_seer_witch_guard")
    args = argparser.parse_args()
    all_prompt_list = []
    all_completion_list = []
    all_label_list = []

    selected_paths = [args.selected_path]
    for selected_path in selected_paths:
        for json_file in tqdm(os.listdir(selected_path)):
            tqdm.write(f"Currently processing: {selected_path}/{json_file}")
            if json_file.endswith(".json"):
                with open(os.path.join(selected_path, json_file), "r") as f:
                    data = json.load(f)

                if "tmp" in json_file or "kto" in json_file:
                    continue
                if "-" in json_file:
                    game_type = json_file.split("-")[0]
                    avs_type = json_file.split("-")[1].replace(".json", "")
                else:
                    game_type = args.game_type
                    avs_type = json_file.replace(".json", "")

                if avs_type in ["villager_bad_vote_loose", "villager_good_vote_strict",
                                "good_speech", "adversarial_good_speech_full", "good_state_recon"]:
                    continue

                if avs_type in ["adversarial_speech_samples", "adversarial_state_samples"]:
                    prompt_list, completion_list, label_list = process_adversarial_sample(data)
                else:
                    prompt_list, completion_list, label_list = process_content_sample(game_type, avs_type, data)
                all_prompt_list.extend(prompt_list)
                all_completion_list.extend(completion_list)
                all_label_list.extend(label_list)
    

    jsonl_files = list(Path(args.conflict_speech_path).glob('*.jsonl'))
    prompt_list, completion_list, label_list = process_good_speech_sample(jsonl_files)
    all_prompt_list.extend(prompt_list)
    all_completion_list.extend(completion_list)
    all_label_list.extend(label_list)

    # post processing: Shuffle
    combined = list(zip(all_prompt_list, all_completion_list, all_label_list))

    random.shuffle(combined)
    all_prompt_list, all_completion_list, all_label_list = zip(*combined)
    all_prompt_list = list(all_prompt_list)
    all_completion_list = list(all_completion_list)
    all_label_list = list(all_label_list)

    # post processing: split train/test
    train_prompt_list = all_prompt_list[:int(len(all_prompt_list) * TRAIN_RATIO)]
    train_completion_list = all_completion_list[:int(len(all_prompt_list) * TRAIN_RATIO)]
    train_label_list = all_label_list[:int(len(all_prompt_list) * TRAIN_RATIO)]

    test_prompt_list = all_prompt_list[int(len(all_prompt_list) * TRAIN_RATIO):]
    test_completion_list = all_completion_list[int(len(all_prompt_list) * TRAIN_RATIO):]
    test_label_list = all_label_list[int(len(all_prompt_list) * TRAIN_RATIO):]

    # save to kto_dataset_train and kto_dataset_test json
    with open(f"{args.save_prefix}_train_avs.json", "w") as f:
        data = {"prompt": train_prompt_list, "completion": train_completion_list, "label": train_label_list}
        json.dump(data, f, indent=4, ensure_ascii=False)
        
    with open(f"{args.save_prefix}_test_avs.json", "w") as f:
        data = {"prompt": test_prompt_list, "completion": test_completion_list, "label": test_label_list}
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print("train:", len(train_prompt_list))
    print("test:", len(test_prompt_list))
