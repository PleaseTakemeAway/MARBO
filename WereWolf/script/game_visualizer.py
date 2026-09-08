import json
import argparse
import os
import gradio as gr
from MARBO.wolf.script.app_modules.presets import small_and_beautiful_theme
import yaml

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
with open(f"{current_dir}/app_modules/custom.css", "r", encoding="utf-8") as f:
    customCSS = f.read()

'''
usage:
python3 game_visualizer.py --game_dir {path to the games} --model_setting {game_setting}
'''


def emojing_roles(role, mode="full"):
    roles = ["Witch", "Guard", "Seer", "Villager", "Werewolf", "Hunter"]
    roles_emoji = ["🧙🏻", "💂", "👁️", "👤", "🐺", "🔫"]
    if mode == "full":
        return role + " " + roles_emoji[roles.index(role)]
    else:
        return roles_emoji[roles.index(role)]


def get_vote_results(vote_detail, id2role_emoji=None):
    s = ""
    for vp, players in vote_detail.items():
        if id2role_emoji is not None:
            all_p = ",".join([f"Player {i}{id2role_emoji[int(i)]} " for i in players])
        else:
            all_p = ",".join([f"Player {i}" for i in players])
        if vp != -1:
            if id2role_emoji is not None:
                s += f"* Vote for **Player {vp}**{id2role_emoji[int(vp)]} (total {len(players)} votes): {all_p}\n"
            else:
                s += f"* Vote for **Player {vp}** (total {len(players)} votes): {all_p}\n"
        else:
            s += f"* Not to vote: {all_p}\n"
    s += "\n"
    return s


def model_jugde(text):
    """Normalize model label text for display."""
    if text is None:
        return "unknown"
    text = str(text).strip()
    if not text:
        return "unknown"
    # Keep the full model name unless it is prefixed with role tags like "w-"/"v-".
    if "-" in text:
        prefix, tail = text.split("-", 1)
        if prefix in {"w", "v"} and tail:
            return tail
    return text


def infer_model_names(log_path, model_setting):
    """
    Try to infer werewolf/villager model names from:
    1) model_setting pattern: w-xxx_vs_v-yyy
    2) sibling config.yaml next to game directories
    """
    if isinstance(model_setting, str) and "_vs_" in model_setting:
        left, right = model_setting.split("_vs_", 1)
        return model_jugde(left), model_jugde(right)

    # Fallback: read experiment config.yaml if present.
    # e.g. .../<exp_root>/game_1/game_log.json -> config at .../<exp_root>/config.yaml
    exp_root = os.path.dirname(os.path.dirname(log_path))
    cfg_path = os.path.join(exp_root, "config.yaml")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            agent_cfg = cfg.get("agent_config", {})
            wolf_cfg = agent_cfg.get("werewolf", {})
            vill_cfg = agent_cfg.get("villager", {})

            wolf_model = (
                (wolf_cfg.get("model_params") or {}).get("llm")
                or wolf_cfg.get("model_type")
                or "werewolf_model"
            )
            vill_model = (
                (vill_cfg.get("model_params") or {}).get("llm")
                or vill_cfg.get("model_type")
                or "villager_model"
            )
            return model_jugde(wolf_model), model_jugde(vill_model)
        except Exception:
            pass

    # Last fallback: use model_setting itself for both sides.
    fallback = model_jugde(model_setting if model_setting is not None else "model")
    return fallback, fallback


def get_role_assignment(log_path, model_setting):
    werewolf_model, human_model = infer_model_names(log_path, model_setting)
    with open(log_path, "r") as f:
        log = json.load(f)
    roles = {}
    id2role_emoji = {}
    text = "# **Role assignments:**\n"
    for i in log:
        event = i["event"]
        if event == "god_view":
            for number in i["content"]:
                if i["content"][number] == "Werewolf":
                    name = werewolf_model
                else:
                    name = human_model
                roles[int(number)] = emojing_roles(i["content"][number]) + ", " + name
                id2role_emoji[int(number)] = emojing_roles(i["content"][number], mode="brief")
                text += f'* Player {number} ({name}): {emojing_roles(i["content"][number])} \n'
        if event == "werewolf_team_info":
            break
    text += "\n"
    return roles, id2role_emoji, text


def process_shoot_summary(content):
    reason = content['context']
    hunter_player = content["player"]
    target_player = content["shoot_player"]
    text = f"**{reason}**\n\n"
    text += "> **Identity labels**: "
    for player_id, label in content["role_prediction"].items():
        text += f"Player {player_id}: {','.join(label)}; "
    text += "\n\n"
    return hunter_player, target_player, text


def load_json_records(path):
    with open(path, encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return []

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
        return []
    except json.JSONDecodeError:
        pass

    records = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def get_note_md(note_log_path):
    note_path = os.listdir(note_log_path)
    note_dict = dict()
    player_note_path = [
        os.path.join(note_log_path, path)
        for path in note_path
        if path != "game_log.json"
        and path.endswith((".json", ".jsonl"))
        and os.path.isfile(os.path.join(note_log_path, path))
    ]
    for player_note in player_note_path:
        current_info = [
            info
            for info in load_json_records(player_note)
            if isinstance(info, dict) and "player_id" in info and "message" in info
        ]
        for single_info in current_info:
            player_id = single_info["player_id"]
            if player_id not in note_dict:
                note_dict[player_id] = {}
        vote_info = [info for info in current_info if "vote" in info["message"]]
        for message in vote_info:
            player_id = message["player_id"]
            note_day = message["message"].split("_")[0]
            if "vote_reason" in message:
                vote_reason = message["vote_reason"] + "\n\n"
            else:
                vote_reason = "No voting reason provided" + "\n\n"
            note_dict[player_id][note_day] = vote_reason
    return note_dict


def find_action_reason(log_path, player, role, day, event):
    action_reason_ret = ""
    phase = f"{day}_night_{event}"
    file = log_path.replace("game_log.json", f"Player_{player}.jsonl")
    for content in load_json_records(file):
        if not isinstance(content, dict) or content.get("phase") != phase:
            continue

        res = content.get("action_reason", "")
        if isinstance(res, list):
            parts = [str(item).strip() for item in res if str(item).strip()]
            if parts:
                action_reason_ret = " ".join(parts)
                break
        elif isinstance(res, str):
            if res.strip():
                action_reason_ret = res.replace("\n", "").strip()
                break
        elif res:
            action_reason_ret = str(res).replace("\n", "").strip()
            break
    return action_reason_ret


def find_speech_template(log_path, player, day, event):
    speech_template_ret = ""
    phase = f"{day}_day_{event}"
    file = log_path.replace("game_log.json", f"Player_{player}.jsonl")
    with open(file, "r") as f:
        for line in f:
            content = json.loads(line.strip())
            if content["phase"] == phase:
                temp = content.get("speech_template", "")
                if len(temp) > 0:
                    speech_template_ret = (
                        temp
                        .replace("贴上身份标签", "")
                        .replace("把", "")
                        .replace("attach identity labels", "")
                    )
                    break
    return speech_template_ret


def format_player_label(player_id, roles):
    if player_id == -1:
        return "Abstain"
    if player_id in roles:
        return f"Player {player_id} ({roles[player_id]})"
    return f"Player {player_id}"


def append_vote_summary_rows(vote_rows, vote_detail, roles, vote_stage, vote_phase, eliminated_target):
    eliminated_label = (
        "No elimination"
        if eliminated_target == 0
        else format_player_label(eliminated_target, roles)
    )
    if not vote_detail:
        vote_rows.append(
            [f"Day {vote_stage}", vote_phase, "-", "0", "-", eliminated_label]
        )
        return

    sorted_vote_items = sorted(
        vote_detail.items(),
        key=lambda item: (item[0] == -1, item[0]),
    )
    for vote_to, voters in sorted_vote_items:
        vote_rows.append(
            [
                f"Day {vote_stage}",
                vote_phase,
                format_player_label(vote_to, roles),
                str(len(voters)),
                ", ".join([f"Player {v}" for v in voters]),
                eliminated_label,
            ]
        )


def get_gamelog_md(roles, id2role_emoji, log_path, model_setting):
    with open(log_path, "r") as f:
        log = json.load(f)
    round_num = 0
    speech_text = ""
    vote_out_players = []
    all_votes_reasonings = {i + 1: {} for i in range(len(roles))}
    all_reviews = {}
    end_flag = False
    current_player_speech = None
    speech_summary_dict = {}
    day_processed = []
    voting_rows = []
    action_rows = []
    vote_phase = "Normal"
    vote_stage = 1
    vote_detail_this_round = {}
    werewolf_night_discussion_text = ""

    werewolf_model, human_model = infer_model_names(log_path, model_setting)
    against = [werewolf_model, human_model]
    for i in log:
        event = i["event"]

        if event == "game_setting":
            round_num = int(i["day"]) + 1
            vote_detail_this_round = {}
            speech_summary_dict = {}
            werewolf_night_discussion_text = ""
            current_player_speech = None
            vote_stage = round_num
            vote_phase = "Normal"

        elif event == "skill_seer":
            action_reason = find_action_reason(log_path, i["source"], role="Seer", day=i["day"], event=i["event"])
            detail_text = f"Checked Player {i['target']}"
            if i["content"]["checked_identity"] == "bad":
                detail_text += " -> werewolf"
            else:
                detail_text += " -> not werewolf"
            action_rows.append(
                [
                    f"Night {int(i['day']) + 1}",
                    "Seer Check",
                    format_player_label(i["source"], roles),
                    format_player_label(i["target"], roles),
                    detail_text,
                    action_reason or "-",
                ]
            )

        elif event == "skill_guard":
            action_reason = find_action_reason(log_path, i["source"], role="Guard", day=i["day"], event=i["event"])
            if i["target"] != 0:
                target_label = format_player_label(i["target"], roles)
                detail_text = "Protected"
            else:
                target_label = "-"
                detail_text = "No protection"
            action_rows.append(
                [
                    f"Night {int(i['day']) + 1}",
                    "Guard Protect",
                    format_player_label(i["source"], roles),
                    target_label,
                    detail_text,
                    action_reason or "-",
                ]
            )

        elif event == "skill_wolf":
            action_reason = find_action_reason(log_path, i["source"], role="Werewolf", day=i["day"], event=i["event"])
            werewolf_night_discussion_text += (
                f"Player {i['source']} chose to kill Player {i['target']} *[{action_reason}]*;\n"
            )
            action_rows.append(
                [
                    f"Night {int(i['day']) + 1}",
                    "Werewolf Attack Vote",
                    format_player_label(i["source"], roles),
                    format_player_label(i["target"], roles),
                    "Chose night kill target",
                    action_reason or "-",
                ]
            )

        elif event == "kill_decision":
            action_rows.append(
                [
                    f"Night {round_num}",
                    "Kill Decision",
                    "Werewolves",
                    format_player_label(i["target"], roles),
                    "Final night kill target",
                    "-",
                ]
            )

        elif event == "skill_witch":
            action_reason = find_action_reason(log_path, i["source"], role="Witch", day=i["day"], event=i["event"])
            if "poison" in i["content"]:
                detail_text = "Poisoned target, no heal"
                target_label = format_player_label(i["target"], roles)
            elif "heal" in i["content"]:
                detail_text = "Healed target, no poison"
                target_label = format_player_label(i["target"], roles)
            elif "pass" in i["content"]:
                detail_text = "No heal, no poison"
                target_label = "-"
            else:
                detail_text = str(i["content"])
                target_label = format_player_label(i.get("target", 0), roles) if i.get("target", 0) else "-"
            action_rows.append(
                [
                    f"Night {int(i['day']) + 1}",
                    "Witch Skill",
                    format_player_label(i["source"], roles),
                    target_label,
                    detail_text,
                    action_reason or "-",
                ]
            )

        elif event == "speech":
            if i["day"] not in day_processed:
                speech_text += f"\n----\n# Day {i['day']} speeches. \n"
                day_processed.append(i["day"])
            werewolf_night_discussion_text = ""
            current_player_speech = i["source"]
            role = roles[current_player_speech]
            speech_text += f"- **Player {i['source']}** ({role}):\n"
            speech_text += f"\n**{i['content']['speech_content']}**\n"

            speech_temp = find_speech_template(log_path, i["source"], i["day"], i["event"])
            if len(speech_temp) > 0:
                speech_text += f"> *{speech_temp}*\n"

            if current_player_speech in speech_summary_dict and len(speech_summary_dict[current_player_speech]) > 0:
                speech_text += speech_summary_dict[current_player_speech]
                speech_summary_dict[current_player_speech] = ""

        elif event == "vote":
            vote_stage = i["day"]
            vote_phase = "Normal"
            p = i["source"]
            vote_to = i["target"]
            if vote_to != 0:
                vote_text = f"Voted for **Player {vote_to}** ({roles[vote_to]}).\n"
            else:
                vote_to = -1
                vote_text = "Abstained.\n"
            all_votes_reasonings[p][f"Day {vote_stage}"] = vote_text
            if vote_to not in vote_detail_this_round:
                vote_detail_this_round[vote_to] = [i["source"]]
            else:
                vote_detail_this_round[vote_to].append(i["source"])

        elif event == "end_vote":
            append_vote_summary_rows(
                voting_rows,
                vote_detail_this_round,
                roles,
                vote_stage,
                vote_phase,
                i["target"],
            )
            round_num = int(i["day"]) + 1
            if i["target"] != 0:
                vote_out_players.append(i["target"])
            vote_detail_this_round = {}
            speech_summary_dict = {}
            werewolf_night_discussion_text = ""
            current_player_speech = None
            vote_stage = round_num
            vote_phase = "Normal"

        elif event == "speech_pk":
            if "Tiebreak PK Phase" not in speech_text:
                speech_text += f"## Day {vote_stage - 1} Tiebreak PK Phase \n"
            current_player_speech = i["source"]
            role = roles[current_player_speech]
            speech_text += f"- **Player {i['source']}** ({role}):\n"
            speech_text += f"\n**{i['content']['speech_content']}**\n"
            if current_player_speech in speech_summary_dict and len(speech_summary_dict[current_player_speech]) > 0:
                speech_text += speech_summary_dict[current_player_speech]
                speech_summary_dict[current_player_speech] = ""

        elif event == "vote_pk":
            vote_stage = i["day"]
            vote_phase = "PK"
            p = i["source"]
            vote_to = i["target"]
            if vote_to != 0:
                vote_text = f"Voted for **Player {vote_to}** ({roles[vote_to]}).\n"
            else:
                vote_to = -1
                vote_text = "Abstained.\n"
            all_votes_reasonings[p][f"Day {vote_stage}_pk"] = vote_text
            if vote_to not in vote_detail_this_round:
                vote_detail_this_round[vote_to] = [i["source"]]
            else:
                vote_detail_this_round[vote_to].append(i["source"])

        elif event == "end_game" and not end_flag:
            if i["content"]["outcome"] == -1:
                winner, symbol = against[-1], "👤"
            else:
                winner, symbol = against[0], "🐺"
            speech_text += f"\n\n## Game end at Round {i['day']}. {symbol}***{winner}*** wins!"
            end_flag = True

    speech_text += "\n"
    return speech_text, all_votes_reasonings, vote_out_players, all_reviews, voting_rows, action_rows


def find_matching_pk(game_dir, model_setting, choice):
    game_record = os.listdir(game_dir)
    if len(choice) == 0:
        # Default: include every game_* directory that has game_log.json
        matching_paths = [
            os.path.join(game_dir, path)
            for path in game_record
            if path.startswith("game_")
            and os.path.isdir(os.path.join(game_dir, path))
            and "game_log.json" in os.listdir(os.path.join(game_dir, path))
        ]
        return sorted(matching_paths)

    matching_paths = [
        os.path.join(game_dir, path)
        for path in game_record
        if path in choice
        and os.path.isdir(os.path.join(game_dir, path))
        and "game_log.json" in os.listdir(os.path.join(game_dir, path))
    ]
    return matching_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--game_dir", type=str, default=None, help="game path")
    parser.add_argument("--model_setting", type=str, default=None,
                        help="game setting: example: w-makto_vs_v-gpt4")
    args = parser.parse_args()

    # choice = ["game_1", "game_2", "game_3", "game_4"]
    choice = []  # if choice is empty, all the games will be checked

    game_ready_list = find_matching_pk(args.game_dir, args.model_setting, choice)
    if args.game_dir is not None and args.model_setting is None:
        args.model_setting = os.path.basename(args.game_dir.rstrip("/"))
    with gr.Blocks(css=customCSS, theme=small_and_beautiful_theme) as demo:
        demo.title = "WereWolf demo"
        gr.Markdown(f"{args.model_setting}\n")
        for game_choice in game_ready_list:
            game_id = game_choice.split("/")[-1]
            with gr.Tab(f"{game_id}"):
                event_log = os.path.join(game_choice, "game_log.json")
                roles, id2role_emoji, role_assignment_md = get_role_assignment(event_log, args.model_setting)
                all_notes = get_note_md(game_choice)
                speech_log_md, all_votes, vote_out_players, all_reviews, voting_rows, action_rows = get_gamelog_md(
                    roles,
                    id2role_emoji,
                    event_log,
                    args.model_setting,
                )
                with gr.Row():
                    gr.Markdown(role_assignment_md)
                with gr.Row():
                    gr.Markdown("## Overall Voting Result")
                with gr.Row():
                    gr.Dataframe(
                        headers=["Day", "Phase", "Voted Target", "Votes", "Voters", "Eliminated"],
                        value=voting_rows,
                        wrap=True,
                        interactive=False,
                    )
                with gr.Row():
                    gr.Markdown("## Action Record")
                with gr.Row():
                    gr.Dataframe(
                        headers=["Night", "Action", "Actor", "Target", "Detail", "Reason"],
                        value=action_rows,
                        wrap=True,
                        interactive=False,
                    )
                with gr.Row():
                    with gr.Column(scale=6):
                        gr.Markdown("# Speech\n" + speech_log_md)
                    with gr.Column(scale=4):
                        gr.Markdown("# Votings & NoteTakings")
                        for player_id in all_votes.keys():
                            note_day_processed = []
                            with gr.Tab(f"Player {player_id} ({roles[player_id]})"):
                                if len(all_votes[player_id]) == 0:
                                    gr.Markdown(f"## Killed on first night 🔪")
                                for day in all_votes[player_id]:
                                    with gr.Row():
                                        gr.Markdown(f"# 📅{day}")
                                    with gr.Row():  
                                        gr.Markdown(f"**VOTE**: " + all_votes[player_id][day] + "\n")
                                    with gr.Row(): 
                                        note_day = day.split(" ")[1].split("_")[0]
                                        if note_day in note_day_processed:
                                            continue
                                        else:
                                            if note_day in all_notes[player_id]:
                                                gr.Markdown \
                                                    (f"# 📒Notes:\n" + "\n " + all_notes[player_id][note_day].replace("\n"
                                                                                                                    ,
                                                                                                                    "\n\n") + "\n------")
                                            elif player_id in vote_out_players: 
                                                gr.Markdown \
                                                    (f"# 📒Notes:\n" + "- **VOTED OUT! No note-taking this turn.**" + "\n------")
                                            note_day_processed.append(note_day)

    demo.queue(max_size=20).launch(server_name="0.0.0.0", share=True)
