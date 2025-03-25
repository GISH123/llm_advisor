# src/advisor.py
import yaml
import logging
import time  # Added for time measurement
from src.model_loader import LLMModelLoader
from src.baccarat_rules import calculate_hand_value, determine_winner
from src.baccarat_stats import calculate_win_probabilities

logger = logging.getLogger(__name__)

class BaccaratLLMAdvisor:
    def __init__(self, config_path):
        """Initialize the advisor with a config file."""
        self.config = self._load_config(config_path)
        if self.config.get("enabled", True):
            self.model_loader = LLMModelLoader(
                self.config["model_path"],
                self.config.get("use_gpu", True)
            )
        else:
            self.model_loader = None

    def _load_config(self, config_path):
        """Load configuration from a YAML file."""
        try:
            with open(config_path, "r", encoding="utf-8") as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}

    def _create_prompt(self, game_state, probabilities):
        player_hand = game_state['player_cards'].split('，')
        player_total = game_state['player_points']
        banker_hand = game_state['banker_cards'].split('，')
        banker_total = game_state['banker_points']

        # Check for natural wins (8 or 9 with two cards, no further draws)
        natural_win = (len(player_hand) == 2 and player_total >= 8) or (len(banker_hand) == 2 and banker_total >= 8)

        if natural_win:
            third_card_info = "不会抽第三张牌"  # No third cards will be drawn
        else:
            # Player draws a third card if total is 0-5 with two cards
            player_draws = len(player_hand) == 2 and player_total <= 5
            if player_draws:
                third_card_info = "玩家将抽第三张牌"  # Player will draw a third card
                # Banker’s draw depends on Player’s third card, but we don’t have it yet in this state
                if len(banker_hand) == 2:
                    third_card_info += "，庄家可能根据玩家的第三张牌抽牌"  # Banker may draw based on Player's third card
            else:
                third_card_info = "玩家停止抽牌"  # Player stops
                # Banker draws if total is 0-5 with two cards and Player didn’t draw
                banker_draws = len(banker_hand) == 2 and banker_total <= 5
                if banker_draws:
                    third_card_info += "，庄家将抽第三张牌"  # Banker will draw a third card
                else:
                    # If Player has 3 cards, check Banker’s draw based on Player’s third card
                    if len(player_hand) == 3 and len(banker_hand) == 2:
                        player_third_card_value = self._get_card_value(player_hand[-1].split()[0])
                        if self._should_banker_draw(banker_total, player_third_card_value):
                            third_card_info += "，庄家将抽第三张牌"  # Banker will draw a third card
                        else:
                            third_card_info += "，庄家停止抽牌"  # Banker stops
                    else:
                        third_card_info += "，庄家停止抽牌"  # Banker stops

        prob_player = probabilities['player'] / 100
        prob_banker = probabilities['banker'] / 100
        prob_tie = probabilities['tie'] / 100

        EV_player = prob_player * 1 + (prob_banker + prob_tie) * (-1)
        EV_banker = prob_banker * 0.95 + (prob_player + prob_tie) * (-1)
        EV_tie = prob_tie * 8 + (prob_player + prob_banker) * (-1)

        prompt = f"""您是一位高度精准的百家乐专家，负责提供实时投注建议，系统已对目前的牌局之后可能的情况，
        按照随机抽牌，以百家乐的规则进行千局模拟，将之后庄家或闲家赢的分布图之概率算出来后，
        放在此 prompt 里面。请考虑系统提供的概率，并考虑标准的百家乐投注赔率：

        - 玩家投注：1:1（平赔）
        - 庄家投注：1:1，赢时收取5%佣金
        - 和局投注：8:1

        当前状态：
        玩家牌：{game_state['player_cards']}（总计 {game_state['player_points']} 点）
        庄家牌：{game_state['banker_cards']}（总计 {game_state['banker_points']} 点）
        游戏当前状态：{third_card_info}

        获胜概率(以千局模拟后的概率)：
        - 玩家：{probabilities['player']:.2f}%
        - 庄家：{probabilities['banker']:.2f}%
        - 和局：{probabilities['tie']:.2f}%

        期望值(概率 * 赔率回报)：
        - 玩家投注：{EV_player:.4f}
        - 庄家投注：{EV_banker:.4f}
        - 和局投注：{EV_tie:.4f}

        请按照步骤详细思考，并于20秒以内回答，基于给定的期望值(应永远选择正期望值最大的选项。如都只有负值的情况下，选择不下注)，
        做出(1)投注玩家(2)投注庄家(3)投注和局(4)不下注
        其中之一的结论。
        =====
        Example:
        期望值(概率 * 赔率回报)：
        - 玩家投注：-0.1114
        - 庄家投注：-0.1315
        - 和局投注：-0.0073
        => 因所有选项皆为负值，选择 (4) 不下注。
        =====

        用中文简要解释您的选择。开头第一句请先阐述下注结论。
        您的建议："""
        return prompt

    # Helper method to extract card value for _should_banker_draw
    def _get_card_value(self, card_str):
        """Extract the numerical value of a card from its string description."""
        value_map = {'A': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 0, 'J': 0, 'Q': 0, 'K': 0}
        return str(value_map.get(card_str, 0))

    def _format_cards_for_prompt(self, resultlist):
        """Format card results into a game state dictionary."""
        player_cards = []
        banker_cards = []

        for card in resultlist:
            card_desc = self._get_card_description(card.classid)
            if card.index in [1, 3, 5]:  # Player positions
                player_cards.append(card_desc)
            elif card.index in [2, 4, 6]:  # Banker positions
                banker_cards.append(card_desc)

        player_points = calculate_hand_value(player_cards)
        banker_points = calculate_hand_value(banker_cards)

        player_text = "，".join(player_cards) if player_cards else "无牌"
        banker_text = "，".join(banker_cards) if banker_cards else "无牌"

        return {
            "player_cards": player_text,
            "player_points": player_points,
            "banker_cards": banker_text,
            "banker_points": banker_points
        }

    def _get_card_description(self, classid):
        """Convert classid to human-readable card description."""
        suits = ["Spade", "Heart", "Diamond", "Club"]
        values = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
        suit_idx = (classid - 1) // 13
        value_idx = (classid - 1) % 13
        return f"{values[value_idx]} {suits[suit_idx]}"

    def _get_fallback_advice(self, resultlist):
        """Return fallback advice if the model is unavailable."""
        return "由于模型未加载，建议根据历史趋势投注。"

    def _should_banker_draw(self, banker_total, player_third_card):
        """Determine if Banker should draw a third card based on Baccarat rules."""
        if banker_total <= 2:
            return True
        if banker_total == 3 and player_third_card != '8':
            return True
        if banker_total == 4 and 2 <= int(player_third_card) <= 7:
            return True
        if banker_total == 5 and 4 <= int(player_third_card) <= 7:
            return True
        if banker_total == 6 and player_third_card in ['6', '7']:
            return True
        return False

    def get_advice(self, gmcode, resultlist):
        """Generate advice using the loaded model or return fallback."""
        if not self.config.get("enabled", True) or \
           self.model_loader is None or \
           self.model_loader.model is None:
            return self._get_fallback_advice(resultlist)

        game_state = self._format_cards_for_prompt(resultlist)
        # Placeholder for remaining deck (empty for now; enhance as needed)
        deck = []

        # Start timing the probability simulation
        sim_start_time = time.time()
        logger.info("Starting 1000-game simulation for probabilities")
        probabilities = calculate_win_probabilities(
            game_state['player_cards'].split('，'),
            game_state['banker_cards'].split('，'),
            deck
        )
        sim_elapsed_time = time.time() - sim_start_time
        logger.info(f"1000-game simulation completed in {sim_elapsed_time:.2f} seconds")

        prompt = self._create_prompt(game_state, probabilities)
        
        try:
            # Start timing the LLM processing
            llm_start_time = time.time()
            
            logger.info("Starting LLM prompt processing")
            inputs = self.model_loader.tokenizer(prompt, return_tensors="pt").to(self.model_loader.device)
            output_ids = self.model_loader.model.generate(
                **inputs,
                max_new_tokens=self.config.get("max_new_tokens", 150)
            )
            advice = self.model_loader.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            # Calculate and log elapsed time for LLM
            llm_elapsed_time = time.time() - llm_start_time
            logger.info(f"LLM prompt processing completed in {llm_elapsed_time:.2f} seconds")
            
            return advice
        except Exception as e:
            logger.error(f"Error generating advice: {e}")
            return self._get_fallback_advice(resultlist)