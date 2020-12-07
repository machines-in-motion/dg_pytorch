/**
 * @file dg_pytorch.hpp
 * @author Julian Viereck
 * @license License BSD-3-Clause
 * @copyright Copyright (c) 2019, New York University and Max Planck Gesellschaft.
 * @date 2020-12-07
 * @brief Pytorch bindings for dynamic graph.
 */

#pragma once

/* --------------------------------------------------------------------- */
/* --- INCLUDE --------------------------------------------------------- */
/* --------------------------------------------------------------------- */

/* SOT */
#include <dynamic-graph/signal-time-dependent.h>
#include <dynamic-graph/signal-ptr.h>
#include <dynamic-graph/entity.h>

#include <torch/script.h> // One-stop header.

#include <real_time_tools/timer.hpp>

namespace dg = dynamicgraph;

/* --------------------------------------------------------------------- */
/* --- API ------------------------------------------------------------- */
/* --------------------------------------------------------------------- */

#if defined (WIN32)
#  if defined (DG_PYTORCH_EXPORTS)
#    define DG_PYTORCH_EXPORTS __declspec(dllexport)
#  else
#    define DG_PYTORCH_EXPORTS  __declspec(dllimport)
#  endif
#else
#  define DG_PYTORCH_EXPORTS
#endif

namespace dg_pytorch {

    /* --------------------------------------------------------------------- */
    /* --- CLASS ----------------------------------------------------------- */
    /* --------------------------------------------------------------------- */

    /**
     * @brief Simple shortcut for code writing convenience
     */
    typedef dynamicgraph::SignalPtr<dynamicgraph::Vector,int> SignalIn;

    /**
     * @brief Simple shortcut for code writing convenience
     */
    typedef dynamicgraph::SignalTimeDependent<dynamicgraph::Vector,int> SignalOut;


    typedef dynamicgraph::SignalTimeDependent<double,int> SignalRefresher;

    /**
     * @brief Entity for running pytorch/torchscript modules from dynamic graph.
     */
    class DG_PYTORCH_EXPORTS PyTorchEntity: public dg::Entity
    {
    protected:
        torch::jit::script::Module module_;

        /**
         * @brief Create an internal output signal which is "ALWAYS_READY", this means
         * that the signals that depends on it will always be evaluated using the
         * callback functino provided.
         */
        SignalRefresher internal_signal_refresher_;

        /**
         * @brief Signal used to run / update the neural network.
         */
        SignalRefresher signal_run_network_;


        std::vector<std::pair<dg::Vector, std::unique_ptr<SignalIn> > > input_signals_;

        std::vector<std::unique_ptr<SignalOut> > output_signals_;
        std::vector<std::string> output_signal_names_;

        torch::jit::IValue network_result_;

        std::vector<torch::jit::IValue> net_inputs_;

        /**
         * @brief Timer for recording the execution time of the network.
         */
        real_time_tools::Timer run_timer_;
    public:
        PyTorchEntity(const std::string & name);

        static const std::string CLASS_NAME;
        virtual const std::string& getClassName( void ) const {return CLASS_NAME;}

        void load_model(const std::string& script_module);

        void add_input(const std::string& signal_name);

        void add_output(const std::string& signal_name);

        void warmup();

        /**
         * @brief Runs / executes the network. Computes the values for the input
         * signals and uses them for the network execution.
         *
         * @returns double& duration in ms required to run the network this time.
         */
        double& run_network(double& res, const int& time);

        /**
         * @brief Callbackk for the registered output signals of the network.
         */
        dynamicgraph::Vector& signal_callbacks(const std::string& tf_op_name,
                                                dynamicgraph::Vector& res,
                                                const int& /*time*/);

    };

}
