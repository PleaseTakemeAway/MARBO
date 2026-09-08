
import json
import os
import argparse
import random
from MARBO.wolf.preference_extraction.utils_english import get_system_prompt
from MARBO.wolf.preference_extraction.utils_english import get_assignment_role, to_text, normalize_role_terms_to_english, resolve_game_path
from tqdm import tqdm
from pathlib import Path
import re

TRAIN_RATIO = 1.0

ROLE_TO_EN = {
    "Villager": "Villager",
    "Werewolf": "Werewolf",
    "Seer": "Seer",
    "Witch": "Witch",
    "Guard": "Guard",
    "Hunter": "Hunter",
    "村民": "Villager",
    "狼人": "Werewolf",
    "预言家": "Seer",
    "女巫": "Witch",
    "守卫": "Guard",
    "猎人": "Hunter",
}

def extract_identity(text):
    match = re.search(r'(?:你的身份是：|Your identity is: )(.+?)(?:。|\.|\n)', text)
    if match:
        role = match.group(1).strip()
        return ROLE_TO_EN.get(role, role)
    else:
        return None
    
def create_message(role, content):
    return {"role": role, "content": content}

def process_content_sample(game_type, avs_type, data):
    prompt_list = []
    completion_list = []
    label_list = []
    role_label_list = []
    phase_list_out = []
    game_path_list = []

    if not isinstance(data, dict):
        try:
            data = dict(data)  # supports dict_items or list of (key, value)
        except Exception:
            return prompt_list, completion_list, label_list, role_label_list, phase_list_out, game_path_list
    if "good" in avs_type:
        label = True
    else:
        label = False
    
    def add_sample(prompt, completion, label, role_label, phase, game_path):
        prompt_list.append(prompt)
        completion_list.append(completion)
        label_list.append(label)
        role_label_list.append(role_label)
        phase_list_out.append(phase)
        game_path_list.append(game_path)

    role_cache = {}
    for file_root, content in data.items():
        file_root_abs = os.path.abspath(file_root)
        for i, (player_id, phase_list) in enumerate(content.items()):
            if not os.path.exists(os.path.join(file_root_abs, f"Player_{player_id}.jsonl")):
                print("not exist", os.path.join(file_root_abs, f"Player_{player_id}.jsonl"))
                continue
            with open(os.path.join(file_root_abs, f"Player_{player_id}.jsonl"), "r") as f:
                for line in f:
                    player_game_play_info = json.loads(line.strip())
                    phase = player_game_play_info["phase"]
                    if phase in phase_list:
                        sample_game_path = resolve_game_path(
                            player_game_play_info.get("game_path", ""),
                            file_root_abs
                        )
                        game_role_label = get_assignment_role(sample_game_path, role_cache)
                        system_prompt = get_system_prompt(game_type)
                        input_prompt = player_game_play_info["prompt"]
                        response = player_game_play_info["response"]
                        if input_prompt is None or response is None or len(response)==0 or len(input_prompt)==0:
                            continue
                        if '\n** 游戏说明\n你现在正在玩一种叫做“狼人杀”的游戏。' in input_prompt \
                           or 'The rules and instructions for the Werewolf game are as follows:' in input_prompt: # filter gpt input
                            continue
                        
                        input_prompt = normalize_role_terms_to_english(input_prompt)
                        prompt = [create_message("system", system_prompt),
                                  create_message("user", input_prompt)]
                        completion = [create_message("assistant", response)]

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
                                    add_sample(prompt, completion, label, game_role_label, phase, sample_game_path)
                            elif "pk" in phase:
                                if random.random() > 0.01:
                                    add_sample(prompt, completion, label, game_role_label, phase, sample_game_path)
                        elif 'good_vote' in avs_type:
                            if any(x in phase for x in ['1_day_vote', '2_day_vote']) and ('pk' not in phase): # downsample
                                if random.random() > 0.4:
                                    continue
                            elif any(x in phase for x in ['3_day_vote', '4_day_vote']) and ('pk' not in phase): # add twice
                                if random.random() > 0.01:
                                    add_sample(prompt, completion, label, game_role_label, phase, sample_game_path)
                            elif 'pk' in phase:
                                add_sample(prompt, completion, label, game_role_label, phase, sample_game_path)
                        elif 'bad_speech' in avs_type:
                            if any(x in phase for x in ['1_day_speech', '2_day_speech']) and ('pk' not in phase): # downsample
                                if random.random() > 0.9:
                                    continue
                            elif any(x in phase for x in ['4_day_speech', '5_day_speech']) and ('pk' not in phase): # add twice
                                if random.random() > 0.01:
                                    add_sample(prompt, completion, label, game_role_label, phase, sample_game_path)
                            elif 'pk' in phase:
                                add_sample(prompt, completion, label, game_role_label, phase, sample_game_path)
                        elif 'good_action' in avs_type:
                            if any(x in phase for x in ['1_night_skill_wolf']): # add twice
                                if random.random() > 0.5:
                                    continue
                            elif any(x in phase for x in ['2_night_skill_witch', '3_night_skill', "4_night_skill"]): # add twice
                                if random.random() > 0.01:
                                    add_sample(prompt, completion, label, game_role_label, phase, sample_game_path)
                                    add_sample(prompt, completion, label, game_role_label, phase, sample_game_path)
                        elif 'bad_action' in avs_type:
                            if random.random() < 0.999:
                                add_sample(prompt, completion, label, game_role_label, phase, sample_game_path)

                        add_sample(prompt, completion, label, game_role_label, phase, sample_game_path)
                        
    return prompt_list, completion_list, label_list, role_label_list, phase_list_out, game_path_list
    

def process_good_speech_sample(jsonl_files):
    prompt_list = []
    completion_list = []
    label_list = []
    role_label_list = []
    phase_list_out = []
    game_path_list = []

    def add_sample(prompt, completion, label, role_label, phase, game_path):
        prompt_list.append(prompt)
        completion_list.append(completion)
        label_list.append(label)
        role_label_list.append(role_label)
        phase_list_out.append(phase)
        game_path_list.append(game_path)

    role_cache = {}
    for file_path in jsonl_files:
        print(f"Processing file: {file_path}")
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                content = json.loads(line.strip())
                prompt_text = to_text(content.get("prompt"))
                response_text = to_text(content.get("response"))
                prompt_text = normalize_role_terms_to_english(prompt_text)
                if len(prompt_text) == 0 or len(response_text) == 0:
                    continue
                role = extract_identity(prompt_text)
                phase = content["phase"]
                sample_game_path = resolve_game_path(content.get("game_path", ""))
                game_role_label = get_assignment_role(sample_game_path, role_cache)
                prompt = [create_message("system", content["system_prompt"]), 
                          create_message("user", prompt_text)]
                completion = [create_message("assistant", response_text)]

                if content["judge_by_llm"]["final_judge"] == "accept":
                    label = True
                    if role == "Villager":
                        if random.random() > 0.4:
                            continue
                    elif role == "Werewolf":
                        if random.random() < 0.4:
                            continue
                    if "pk" in phase:
                        if random.random() > 0.05:
                            add_sample(prompt, completion, label, game_role_label, phase, sample_game_path)
                    elif phase == "1_day_speech":
                        if random.random() > 0.5:
                            continue
                    # elif any(x in phase for x in ['2_day_speech', '2_day_speech']):
                    #     if random.random() > 0.5:
                    #         continue
                    elif any(x in phase for x in ['3_day_speech', '4_day_speech', '5_day_speech']):
                        if random.random() > 0.2:
                            add_sample(prompt, completion, label, game_role_label, phase, sample_game_path)
                else:
                    label = False
                    if role == "Seer":
                        add_sample(prompt, completion, label, game_role_label, phase, sample_game_path)
                        add_sample(prompt, completion, label, game_role_label, phase, sample_game_path)
                    elif role == "Werewolf":
                        if random.random() < 0.6:
                            continue
                    elif role == "Villager":
                        if random.random() < 0.4:
                            continue
                    if 'pk' in phase:
                        add_sample(prompt, completion, label, game_role_label, phase, sample_game_path)
                        add_sample(prompt, completion, label, game_role_label, phase, sample_game_path)
                    elif any(x in phase for x in ['1_day_speech', '2_day_speech']):
                        if random.random() > 0.7:
                            continue
                    elif any(x in phase for x in ['3_day_speech', '4_day_speech', '5_day_speech']):
                        if random.random() > 0.01:
                            add_sample(prompt, completion, label, game_role_label, phase, sample_game_path)
                
                add_sample(prompt, completion, label, game_role_label, phase, sample_game_path)
    return prompt_list, completion_list, label_list, role_label_list, phase_list_out, game_path_list

def process_adversarial_sample(samples):
    """
    Process adversarial_speech_samples.json and adversarial_state_samples.json.
    Each file is a JSON array where each entry has:
      prompt, completion, label, role_label, phase, game_path, player_id
    completion may be a list of messages (speech) or a dict (state_recon).
    """
    prompt_list = []
    completion_list = []
    label_list = []
    role_label_list = []
    phase_list_out = []
    game_path_list = []

    for sample in samples:
        prompt = sample.get("prompt")
        completion = sample.get("completion")
        label = sample.get("label")
        role_label = sample.get("role_label", {})
        phase = sample.get("phase", "")
        game_path = resolve_game_path(sample.get("game_path", ""))

        if prompt is None or completion is None or label is None:
            continue
        if not isinstance(prompt, list) or len(prompt) == 0:
            continue

        # Normalize completion to list-of-messages format
        if isinstance(completion, dict):
            completion = [create_message("assistant", json.dumps(completion, ensure_ascii=False))]
        elif isinstance(completion, list):
            pass  # already in correct format
        else:
            completion = [create_message("assistant", str(completion))]

        prompt_list.append(prompt)
        completion_list.append(completion)
        label_list.append(label)
        role_label_list.append(role_label)
        phase_list_out.append(phase)
        game_path_list.append(game_path)

    return prompt_list, completion_list, label_list, role_label_list, phase_list_out, game_path_list


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
    all_role_label_list = []
    all_phase_list = []
    all_game_path_list = []

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

                if avs_type in ["villager_bad_vote_loose", "villager_bad_vote_strict",
                                "villager_good_vote_loose", "villager_good_vote_strict",
                                "adversarial_good_speech_full"]:
                    continue

                if avs_type in ["adversarial_speech_samples", "adversarial_state_samples"]:
                    prompt_list, completion_list, label_list, role_label_list, phase_list, game_path_list = process_adversarial_sample(data)
                else:
                    prompt_list, completion_list, label_list, role_label_list, phase_list, game_path_list = process_content_sample(game_type, avs_type, data)
                all_prompt_list.extend(prompt_list)
                all_completion_list.extend(completion_list)
                all_label_list.extend(label_list)
                all_role_label_list.extend(role_label_list)
                all_phase_list.extend(phase_list)
                all_game_path_list.extend(game_path_list)
    
    # process "good" but conflict speech
    jsonl_files = list(Path(args.conflict_speech_path).glob('*.jsonl'))
    prompt_list, completion_list, label_list, role_label_list, phase_list, game_path_list = process_good_speech_sample(jsonl_files)
    all_prompt_list.extend(prompt_list)
    all_completion_list.extend(completion_list)
    all_label_list.extend(label_list)
    all_role_label_list.extend(role_label_list)
    all_phase_list.extend(phase_list)
    all_game_path_list.extend(game_path_list)

    # post processing: Shuffle
    combined = list(zip(all_prompt_list, all_completion_list, all_label_list, all_role_label_list, all_phase_list, all_game_path_list))
    random.shuffle(combined)
    all_prompt_list, all_completion_list, all_label_list, all_role_label_list, all_phase_list, all_game_path_list = zip(*combined)
    all_prompt_list = list(all_prompt_list)
    all_completion_list = list(all_completion_list)
    all_label_list = list(all_label_list)
    all_role_label_list = list(all_role_label_list)
    all_phase_list = list(all_phase_list)
    all_game_path_list = list(all_game_path_list)

    # post processing: split train/test
    train_prompt_list = all_prompt_list[:int(len(all_prompt_list) * TRAIN_RATIO)]
    train_completion_list = all_completion_list[:int(len(all_prompt_list) * TRAIN_RATIO)]
    train_label_list = all_label_list[:int(len(all_prompt_list) * TRAIN_RATIO)]
    train_role_label_list = all_role_label_list[:int(len(all_prompt_list) * TRAIN_RATIO)]
    train_phase_list = all_phase_list[:int(len(all_prompt_list) * TRAIN_RATIO)]
    train_game_path_list = all_game_path_list[:int(len(all_prompt_list) * TRAIN_RATIO)]

    test_prompt_list = all_prompt_list[int(len(all_prompt_list) * TRAIN_RATIO):]
    test_completion_list = all_completion_list[int(len(all_prompt_list) * TRAIN_RATIO):]
    test_label_list = all_label_list[int(len(all_prompt_list) * TRAIN_RATIO):]
    test_role_label_list = all_role_label_list[int(len(all_prompt_list) * TRAIN_RATIO):]
    test_phase_list = all_phase_list[int(len(all_prompt_list) * TRAIN_RATIO):]
    test_game_path_list = all_game_path_list[int(len(all_prompt_list) * TRAIN_RATIO):]

    # save to kto_dataset_train and kto_dataset_test json
    with open(f"{args.save_prefix}_train.json", "w") as f:
        data = {
            "prompt": train_prompt_list,
            "completion": train_completion_list,
            "label": train_label_list,
            "role_label": train_role_label_list,
            "phase": train_phase_list,
            "game_path": train_game_path_list
        }
        json.dump(data, f, indent=4, ensure_ascii=False)

    # Also save sample-wise format for easier full-sample inspection.
    train_samples = []
    for i in range(len(train_prompt_list)):
        train_samples.append({
            "prompt": train_prompt_list[i],
            "completion": train_completion_list[i],
            "label": train_label_list[i],
            "role_label": train_role_label_list[i],
            "phase": train_phase_list[i],
            "game_path": train_game_path_list[i]
        })
    with open(f"{args.save_prefix}_train_samples.json", "w") as f:
        json.dump(train_samples, f, indent=4, ensure_ascii=False)
        
    with open(f"{args.save_prefix}_test.json", "w") as f:
        data = {
            "prompt": test_prompt_list,
            "completion": test_completion_list,
            "label": test_label_list,
            "role_label": test_role_label_list,
            "phase": test_phase_list,
            "game_path": test_game_path_list
        }
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print("train:", len(train_prompt_list))
    print("test:", len(test_prompt_list))
