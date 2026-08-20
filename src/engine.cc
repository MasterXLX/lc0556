/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2019 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA cuDNN library (or a modified version of those
  libraries), containing parts covered by the terms of the respective
  license agreement, the licensors of this Program grant you additional
  permission to convey the resulting work.
*/

#include "engine.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>

#include "mcts/node.h"
#include "mcts/search.h"
#include "mcts/stoppers/factory.h"
#include "utils/commandline.h"
#include "utils/configfile.h"
#include "utils/fastmath.h"
#include "utils/logging.h"

namespace lczero {
namespace {
const int kDefaultThreads = 2;

const OptionId kThreadsOptionId{"threads", "Threads",
                                "Number of (CPU) worker threads to use.", 't'};
const OptionId kLogFileId{"logfile", "LogFile",
                          "Write log to that file. Special value <stderr> to "
                          "output the log to the console.",
                          'l'};
const OptionId kSyzygyTablebaseId{
    "syzygy-paths", "SyzygyPath",
    "List of Syzygy tablebase directories, list entries separated by system "
    "separator (\";\" for Windows, \":\" for Linux).",
    's'};
const OptionId kPonderId{"", "Ponder",
                         "This option is ignored. Here to please chess GUIs."};
const OptionId kUciChess960{
    "chess960", "UCI_Chess960",
    "Castling moves are encoded as \"king takes rook\"."};
const OptionId kShowWDL{"show-wdl", "UCI_ShowWDL",
                        "Show win, draw and lose probability."};
const OptionId kShowMovesleft{"show-movesleft", "UCI_ShowMovesLeft",
                              "Show estimated moves left."};
const OptionId kStrictUciTiming{"strict-uci-timing", "StrictTiming",
                                "The UCI host compensates for lag, waits for "
                                "the 'readyok' reply before sending 'go' and "
                                "only then starts timing."};
const OptionId kPreload{"preload", "",
                        "Initialize backend and load net on engine startup."};
const OptionId kValueOnly{
    "value-only", "ValueOnly",
    "In value only mode all search parameters are ignored and the position is "
    "evaluated by getting the valuation of every child position and choosing "
    "the worst for the opponent."};
const OptionId kPolicyMix{
    "policy-mix", "PolicyMix",
    "Amount to mix policy into the value in value-only mode."};

MoveList StringsToMovelist(const std::vector<std::string>& moves,
                           const ChessBoard& board) {
  MoveList result;
  if (moves.size()) {
    result.reserve(moves.size());
    const auto legal_moves = board.GenerateLegalMoves();
    const auto end = legal_moves.end();
    for (const auto& move : moves) {
      const auto m = board.GetModernMove({move, board.flipped()});
      if (std::find(legal_moves.begin(), end, m) != end) result.emplace_back(m);
    }
    if (result.empty()) throw Exception("No legal searchmoves.");
  }
  return result;
}

inline float ComputeWeight(const SearchParams& params, float uncertainty) {
  const float minimum = params.GetUncertaintyWeightingMinimum();
  const float alpha = params.GetUncertaintyWeightingAlpha();
  const float beta = params.GetUncertaintyWeightingBeta();
  return fmin(minimum, alpha * pow(uncertainty, beta));
}

}  // namespace

EngineController::EngineController(std::unique_ptr<UciResponder> uci_responder,
                                   const OptionsDict& options)
    : options_(options),
      uci_responder_(std::move(uci_responder)),
      current_position_{ChessBoard::kStartposFen, {}} {}

void EngineController::PopulateOptions(OptionsParser* options) {
  using namespace std::placeholders;
  const bool is_simple =
      CommandLine::BinaryName().find("simple") != std::string::npos;
  NetworkFactory::PopulateOptions(options);
  options->Add<IntOption>(kThreadsOptionId, 1, 128) = kDefaultThreads;
  options->Add<IntOption>(kNNCacheSizeId, 0, 999999999) = 2000000;
  SearchParams::Populate(options);

  ConfigFile::PopulateOptions(options);
  if (is_simple) {
    options->HideAllOptions();
    options->UnhideOption(kThreadsOptionId);
    options->UnhideOption(NetworkFactory::kWeightsId);
    options->UnhideOption(SearchParams::kContemptId);
    options->UnhideOption(SearchParams::kMultiPvId);
  }
  options->Add<StringOption>(kSyzygyTablebaseId);
  // Add "Ponder" option to signal to GUIs that we support pondering.
  // This option is currently not used by lc0 in any way.
  options->Add<BoolOption>(kPonderId) = true;
  options->Add<BoolOption>(kUciChess960) = false;
  options->Add<BoolOption>(kShowWDL) = false;
  options->Add<BoolOption>(kShowMovesleft) = false;

  PopulateTimeManagementOptions(is_simple ? RunType::kSimpleUci : RunType::kUci,
                                options);

  options->Add<BoolOption>(kStrictUciTiming) = false;
  options->HideOption(kStrictUciTiming);

  options->Add<BoolOption>(kPreload) = false;
  options->Add<BoolOption>(kValueOnly) = false;
  options->Add<FloatOption>(kPolicyMix, -2.0f, 2.0f) = 0.0f;
}

void EngineController::ResetMoveTimer() {
  move_start_time_ = std::chrono::steady_clock::now();
}

// Updates values from Uci options.
void EngineController::UpdateFromUciOptions() {
  SharedLock lock(busy_mutex_);
  if (options_.Get<bool>(kValueOnly)) {
    options_.Set(SearchParams::kCpuctId, 1.0f);
  }
  time_manager_ = MakeTimeManager(options_);
}

void EngineController::Go(const GoParams& params) {
  if (options_.Get<bool>(kValueOnly)) {
    ValueOnlyGo(tree_.get(), network_.get(), options_,
                std::move(uci_responder_));
    return;
  }

  search_ = std::make_unique<Search>(tree_.get(), network_.get(), options_,
                                     std::move(uci_responder_), params);
  search_->StartClock(params);
}

void EngineController::SetPosition(const std::string& fen,
                                   const std::vector<std::string>& moves_str) {
  // Some UCI hosts just call position then immediately call go, while starting
  // the clock on calling 'position'.
  ResetMoveTimer();
  SharedLock lock(busy_mutex_);
  current_position_ = CurrentPosition{fen, moves_str};
  search_.reset();
}

Position EngineController::ApplyPositionMoves() {
  ChessBoard board;
  int no_capture_ply;
  int game_move;
  board.SetFromFen(current_position_.fen, &no_capture_ply, &game_move);
  int game_ply = 2 * game_move - (board.flipped() ? 1 : 2);
  Position pos(board, no_capture_ply, game_ply);
  for (std::string move_str : current_position_.moves) {
    Move move(move_str);
    if (pos.IsBlackToMove()) move.Mirror();
    pos = Position(pos, move);
  }
  return pos;
}

void EngineController::SetupPosition(
    const std::string& fen, const std::vector<std::string>& moves_str) {
  SharedLock lock(busy_mutex_);
  search_.reset();

  UpdateFromUciOptions();

  if (!tree_) tree_ = std::make_unique<NodeTree>(options_);

  std::vector<Move> moves;
  for (const auto& move : moves_str) moves.emplace_back(move);
  const bool is_same_game = tree_->ResetToPosition(fen, moves);
  if (!is_same_game) CreateFreshTimeManager();
}

void EngineController::CreateFreshTimeManager() {
  time_manager_ = MakeTimeManager(options_);
}

namespace {

class PonderResponseTransformer : public TransformingUciResponder {
 public:
  PonderResponseTransformer(std::unique_ptr<UciResponder> parent,
                            std::string ponder_move)
      : TransformingUciResponder(std::move(parent)),
        ponder_move_(std::move(ponder_move)) {}

  void TransformThinkingInfo(std::vector<ThinkingInfo>* infos) override {
    // Output all stats from main variation (not necessary the ponder move)
    // but PV only from ponder move.
    ThinkingInfo ponder_info;
    for (const auto& info : *infos) {
      if (info.multipv <= 1) {
        ponder_info = info;
        if (ponder_info.mate) ponder_info.mate = -*ponder_info.mate;
        if (ponder_info.score) ponder_info.score = -*ponder_info.score;
        if (ponder_info.depth > 1) ponder_info.depth--;
        if (ponder_info.seldepth > 1) ponder_info.seldepth--;
        if (ponder_info.wdl) std::swap(ponder_info.wdl->w, ponder_info.wdl->l);
        ponder_info.pv.clear();
      }
      if (!info.pv.empty() && info.pv[0].as_string() == ponder_move_) {
        ponder_info.pv.assign(info.pv.begin() + 1, info.pv.end());
      }
    }
    infos->clear();
    infos->push_back(ponder_info);
  }

 private:
  std::string ponder_move_;
};

void ValueOnlyGo(NodeTree* tree, Network* network, const OptionsDict& options,
                 std::unique_ptr<UciResponder> responder) {
  const auto input_format = network->GetCapabilities().input_format;

  const auto& board = tree->GetPositionHistory().Last().GetBoard();
  const auto legal_moves = board.GenerateLegalMoves();

  if (!tree->GetCurrentHead()->GetLowNode()) {
    const auto hash = tree->GetHistoryHash(tree->GetPositionHistory());
    auto [low_node, is_miss] = tree->TTGetOrCreate(hash);

    if (!low_node->HasChildren() && !legal_moves.empty()) {
      NNEval eval;
      eval.num_edges = static_cast<uint8_t>(legal_moves.size());
      eval.edges = Edge::FromMovelist(legal_moves);
      low_node->SetNNEval(&eval);
    }

    tree->GetCurrentHead()->SetLowNode(low_node);
  }

  PositionHistory history = tree->GetPositionHistory();

  std::vector<InputPlanes> planes;
  std::vector<float> comp_uncertainty;
  int transform;

  // Sample 0 is the current/root position, used for policy.
  planes.emplace_back(EncodePositionForNN(
      input_format, history, 8, FillEmptyHistory::FEN_ONLY, &transform));

  // Samples 1..N are the non-terminal child positions, used for Q.
  for (auto edge : tree->GetCurrentHead()->Edges()) {
    history.Append(edge.GetMove());

    if (history.ComputeGameResult() == GameResult::UNDECIDED) {
      planes.emplace_back(EncodePositionForNN(
          input_format, history, 8, FillEmptyHistory::FEN_ONLY, nullptr));
      comp_uncertainty.push_back(edge.GetWeight());
    }

    history.Pop();
  }

  int batch_size = options.Get<int>(SearchParams::kMiniBatchSizeId);
  if (batch_size == 0) {
    batch_size = network->GetMiniBatchSize();
  }

  std::vector<float> comp_q;
  std::vector<float> pol;

  bool policy_done = false;
  float max_p = std::numeric_limits<float>::lowest();

  for (size_t i = 0; i < planes.size(); i += batch_size) {
    auto comp = network->NewComputation();

    for (int j = 0; j < batch_size && i + j < planes.size(); ++j) {
      comp->AddInput(std::move(planes[i + j]));
    }

    comp->ComputeBlocking();

    const int actual_batch_size = comp->GetBatchSize();

    int start = 0;

    // The first NN sample is the root position. Get policy from it once.
    if (!policy_done) {
      for (auto edge : tree->GetCurrentHead()->Edges()) {
        const float p =
            comp->GetPVal(0, edge.GetMove().as_nn_index(transform));

        pol.push_back(p);
        if (p > max_p) {
          max_p = p;
        }
      }

      start = 1;
      policy_done = true;
    }

    // Remaining samples in this computation are child positions.
    for (int j = start; j < actual_batch_size; ++j) {
      comp_q.push_back(comp->GetQVal(j));
    }
  }

  const float policy_temperature =
      options.Get<float>(SearchParams::kPolicySoftmaxTempId);

  float sum = 0.0f;

  for (size_t i = 0; i < pol.size(); ++i) {
    pol[i] = FastExp((pol[i] - max_p) / policy_temperature);
    sum += pol[i];
  }

  Move best;
  int comp_idx = 0;
  int polidx = 0;
  float max_q = std::numeric_limits<float>::lowest();

  const SearchParams params(options);

  for (auto edge : tree->GetCurrentHead()->Edges()) {
    history.Append(edge.GetMove());

    const auto result = history.ComputeGameResult();

    float q = -1.0f;

    if (result == GameResult::UNDECIDED) {
      // NN eval is from the side-to-move perspective, so if the child
      // position is good for the opponent, it is bad for us.
      q = -comp_q[comp_idx];
      q /= ComputeWeight(params, comp_uncertainty[comp_idx]);
      ++comp_idx;
    } else if (result == GameResult::DRAW) {
      q = 0.0f;
    } else {
      // A legal move to a non-drawn terminal position without tablebases
      // must be a win.
      q = 1.0f;
    }

    q += (pol[polidx] / sum) *
         options.Get<float>(kPolicyMix);

    if (q >= max_q) {
      max_q = q;
      best = edge.GetMove(
          tree->GetPositionHistory().IsBlackToMove());
    }

    history.Pop();
    ++polidx;
  }

  std::vector<ThinkingInfo> infos;
  ThinkingInfo thinking;
  thinking.depth = 1;
  thinking.seldepth = 1;
  thinking.score = max_q;
  thinking.pv.push_back(best);
  infos.push_back(thinking);
  responder->SendThinkingInfo(infos);
  responder->SendBestMove(best);
}

