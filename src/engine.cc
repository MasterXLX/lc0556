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
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
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

  // Syzygy tablebases.
  std::string tb_paths = options_.Get<std::string>(kSyzygyTablebaseId);
  if (!tb_paths.empty() && tb_paths != tb_paths_) {
    syzygy_tb_ = std::make_unique<SyzygyTablebase>();
    CERR << "Loading Syzygy tablebases from " << tb_paths;
    if (!syzygy_tb_->init(tb_paths)) {
      CERR << "Failed to load Syzygy tablebases!";
      syzygy_tb_ = nullptr;
    }
    tb_paths_ = tb_paths;
  } else if (tb_paths.empty()) {
    syzygy_tb_ = nullptr;
    tb_paths_.clear();
  }

  // Network.
  const auto network_configuration =
      NetworkFactory::BackendConfiguration(options_);
  if (network_configuration_ != network_configuration) {
    network_ = NetworkFactory::LoadNetwork(options_);
    network_configuration_ = network_configuration;
  }

  // Cache size.
  cache_.SetCapacity(options_.Get<int>(kNNCacheSizeId));

  // Check whether we can update the move timer in "Go".
  strict_uci_timing_ = options_.Get<bool>(kStrictUciTiming);
}

void EngineController::EnsureReady() {
  std::unique_lock<RpSharedMutex> lock(busy_mutex_);
  // If a UCI host is waiting for our ready response, we can consider the move
  // not started until we're done ensuring ready.
  ResetMoveTimer();
}

void EngineController::NewGame() {
  // In case anything relies upon defaulting to default position and just calls
  // newgame and goes straight into go.
  ResetMoveTimer();
  SharedLock lock(busy_mutex_);
  cache_.Clear();
  search_.reset();
  tree_.reset();
  CreateFreshTimeManager();
  current_position_ = {ChessBoard::kStartposFen, {}};
  UpdateFromUciOptions();
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

  if (legal_moves.empty()) {
    return;
  }

  PositionHistory history = tree->GetPositionHistory();

  // legal_moves is the authoritative move list for this position.
  // ValueOnly must not use the tree's LowNode/Edges() as a move source.
  std::vector<InputPlanes> planes;
  std::vector<int> sample_to_move_index;

  int transform;

  // Sample 0 is the root position and is used to obtain policy.
  planes.emplace_back(EncodePositionForNN(
      input_format, history, 8, FillEmptyHistory::FEN_ONLY, &transform));
  sample_to_move_index.push_back(-1);

  // Samples 1..N are non-terminal child positions. Keep an explicit
  // mapping from every NN sample back to its legal move index.
  for (size_t move_idx = 0; move_idx < legal_moves.size(); ++move_idx) {
    history.Append(legal_moves[move_idx]);

    if (history.ComputeGameResult() == GameResult::UNDECIDED) {
      planes.emplace_back(EncodePositionForNN(
          input_format, history, 8, FillEmptyHistory::FEN_ONLY, nullptr));
      sample_to_move_index.push_back(static_cast<int>(move_idx));
    }

    history.Pop();
  }

  int batch_size = options.Get<int>(SearchParams::kMiniBatchSizeId);
  if (batch_size == 0) {
    batch_size = network->GetMiniBatchSize();
  }

  std::vector<float> q_by_move(legal_moves.size(), 0.0f);
  std::vector<float> uncertainty_by_move(legal_moves.size(), 0.0f);
  std::vector<bool> has_nn_value(legal_moves.size(), false);

  std::vector<float> pol(legal_moves.size());
  float max_p = std::numeric_limits<float>::lowest();

  for (size_t i = 0; i < planes.size(); i += batch_size) {
    auto comp = network->NewComputation();

    const size_t end = std::min(
        i + static_cast<size_t>(batch_size), planes.size());

    for (size_t j = i; j < end; ++j) {
      comp->AddInput(std::move(planes[j]));
    }

    comp->ComputeBlocking();

    const int actual_batch_size = comp->GetBatchSize();

    // Sample 0 is always the root position. It is only used for policy.
    if (i == 0) {
      for (size_t move_idx = 0; move_idx < legal_moves.size();
           ++move_idx) {
        const float p =
            comp->GetPVal(0, legal_moves[move_idx].as_nn_index(transform));

        pol[move_idx] = p;
        max_p = std::max(max_p, p);
      }
    }

    // Explicitly associate each NN result with its legal move rather than
    // relying on a separate comp_idx that can drift from the edge list.
    for (int j = 0; j < actual_batch_size; ++j) {
      const size_t sample_idx = i + static_cast<size_t>(j);

      if (sample_idx == 0) {
        continue;
      }

      const int move_idx = sample_to_move_index[sample_idx];

      if (move_idx < 0 ||
          static_cast<size_t>(move_idx) >= legal_moves.size()) {
        continue;
      }

      q_by_move[move_idx] = comp->GetQVal(j);
      uncertainty_by_move[move_idx] = comp->GetEVal(j);
      has_nn_value[move_idx] = true;
    }
  }

  const float policy_temperature =
      options.Get<float>(SearchParams::kPolicySoftmaxTempId);

  float sum = 0.0f;

  for (float& p : pol) {
    p = FastExp((p - max_p) / policy_temperature);
    sum += p;
  }

  const SearchParams params(options);
  const float cap = params.GetUncertaintyWeightingCap();
  const float coefficient =
      params.GetUncertaintyWeightingCoefficient();
  const float exponent = params.GetUncertaintyWeightingExponent();

  const bool root_is_black =
      tree->GetPositionHistory().IsBlackToMove();

  Move best;
  float max_q = std::numeric_limits<float>::lowest();

  // Iterate ONLY over legal_moves. No tree edges are consulted here.
  for (size_t move_idx = 0; move_idx < legal_moves.size(); ++move_idx) {
    history.Append(legal_moves[move_idx]);

    const auto result = history.ComputeGameResult();

    float q;

    if (result == GameResult::UNDECIDED) {
      // NN evaluation is from the child side-to-move perspective, so
      // negate it to obtain the value from the root side's perspective.
      q = -q_by_move[move_idx];

      const float uncertainty = uncertainty_by_move[move_idx];

      q /= uncertainty * coefficient + 1 - coefficient / 2;
    } else if (result == GameResult::DRAW) {
      q = 0.0f;
    } else {
      // Convert the terminal result to the root player's perspective.
      const bool root_won =
          root_is_black ? (result == GameResult::BLACK_WON)
                        : (result == GameResult::WHITE_WON);

      q = root_won ? 1.0f : -1.0f;
    }

    if (sum > 0.0f) {
      q += (pol[move_idx] / sum) *
           options.Get<float>(kPolicyMix);
    }

    if (q >= max_q) {
      max_q = q;
      best = legal_moves[move_idx];
    }

    history.Pop();
  }

  // Belt-and-suspenders check: best must have originated from the
  // authoritative legal move list.
  const bool best_is_legal =
      std::find(legal_moves.begin(), legal_moves.end(), best) !=
      legal_moves.end();

  if (!best_is_legal) {
    LOGERR << "ValueOnlyGo selected an illegal move.";
    return;
  }

  std::vector<ThinkingInfo> infos;
  ThinkingInfo thinking;
  thinking.depth = 1;
  infos.push_back(thinking);

  responder->OutputThinkingInfo(&infos);

  BestMoveInfo info(
      best, tree->GetPositionHistory().IsBlackToMove());
  responder->OutputBestMove(&info);
}

void EngineController::Go(const GoParams& params) {
  // TODO: should consecutive calls to go be considered to be a continuation and
  // hence have the same start time like this behaves, or should we check start
  // time hasn't changed since last call to go and capture the new start time
  // now?
  if (strict_uci_timing_ || !move_start_time_) ResetMoveTimer();
  go_params_ = params;

  std::unique_ptr<UciResponder> responder =
      std::make_unique<NonOwningUciRespondForwarder>(uci_responder_.get());

  // Setting up current position, now that it's known whether it's ponder or
  // not.
  if (params.ponder && !current_position_.moves.empty()) {
    std::vector<std::string> moves(current_position_.moves);
    std::string ponder_move = moves.back();
    moves.pop_back();
    SetupPosition(current_position_.fen, moves);
    responder = std::make_unique<PonderResponseTransformer>(
        std::move(responder), ponder_move);
  } else {
    SetupPosition(current_position_.fen, current_position_.moves);
  }

  if (!options_.Get<bool>(kUciChess960)) {
    // Remap FRC castling to legacy castling.
    responder = std::make_unique<Chess960Transformer>(
        std::move(responder), tree_->HeadPosition().GetBoard());
  }

  if (!options_.Get<bool>(kShowWDL)) {
    // Strip WDL information from the response.
    responder = std::make_unique<WDLResponseFilter>(std::move(responder));
  }

  if (!options_.Get<bool>(kShowMovesleft)) {
    // Strip movesleft information from the response.
    responder = std::make_unique<MovesLeftResponseFilter>(std::move(responder));
  }
  if (options_.Get<bool>(kValueOnly)) {
    ValueOnlyGo(tree_.get(), network_.get(), options_, std::move(responder));
    return;
  }

  auto stopper = time_manager_->GetStopper(params, *tree_.get());
  search_ = std::make_unique<Search>(
      tree_.get(), network_.get(), std::move(responder),
      StringsToMovelist(params.searchmoves, tree_->HeadPosition().GetBoard()),
      *move_start_time_, std::move(stopper), params.infinite, params.ponder,
      options_, &cache_, syzygy_tb_.get());

  LOGFILE << "Timer started at "
          << FormatTime(SteadyClockToSystemClock(*move_start_time_));
  search_->StartThreads(options_.Get<int>(kThreadsOptionId));
}

void EngineController::PonderHit() {
  ResetMoveTimer();
  go_params_.ponder = false;
  Go(go_params_);
}

void EngineController::Stop() {
  if (search_) search_->Stop();
}

EngineLoop::EngineLoop()
    : engine_(
          std::make_unique<CallbackUciResponder>(
              std::bind(&UciLoop::SendBestMove, this, std::placeholders::_1),
              std::bind(&UciLoop::SendInfo, this, std::placeholders::_1)),
          options_.GetOptionsDict()) {
  engine_.PopulateOptions(&options_);
  options_.Add<StringOption>(kLogFileId);
}

void EngineLoop::RunLoop() {
  if (!ConfigFile::Init() || !options_.ProcessAllFlags()) return;
  const auto options = options_.GetOptionsDict();
  Logging::Get().SetFilename(options.Get<std::string>(kLogFileId));
  if (options.Get<bool>(kPreload)) engine_.NewGame();
  UciLoop::RunLoop();
}

void EngineLoop::CmdUci() {
  SendId();
  for (const auto& option : options_.ListOptionsUci()) {
    SendResponse(option);
  }
  SendResponse("uciok");
}

void EngineLoop::CmdIsReady() {
  engine_.EnsureReady();
  SendResponse("readyok");
}

void EngineLoop::CmdSetOption(const std::string& name, const std::string& value,
                              const std::string& context) {
  options_.SetUciOption(name, value, context);
  // Set the log filename for the case it was set in UCI option.
  Logging::Get().SetFilename(
      options_.GetOptionsDict().Get<std::string>(kLogFileId));
}

void EngineLoop::CmdUciNewGame() { engine_.NewGame(); }

void EngineLoop::CmdPosition(const std::string& position,
                             const std::vector<std::string>& moves) {
  std::string fen = position;
  if (fen.empty()) {
    fen = ChessBoard::kStartposFen;
  }
  engine_.SetPosition(fen, moves);
}

void EngineLoop::CmdFen() {
  std::string fen = GetFen(engine_.ApplyPositionMoves());
  return SendResponse(fen);
}
void EngineLoop::CmdGo(const GoParams& params) { engine_.Go(params); }

void EngineLoop::CmdPonderHit() { engine_.PonderHit(); }

void EngineLoop::CmdStop() { engine_.Stop(); }

}  // namespace lczero
