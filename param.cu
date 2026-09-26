#include <CLI/CLI.hpp>
#include <cstdlib>

#include "rxmesh/rxmesh_static.h"

#include "rxmesh/algo/tutte_embedding.h"

#include "rxmesh/diff/diff_scalar_problem.h"
#include "rxmesh/diff/gradient_descent.h"
#include "rxmesh/util/log.h"


struct arg
{
    std::string obj_file_name = STRINGIFY(INPUT_DIR) "bunnyhead.obj";
    std::string output_folder = STRINGIFY(OUTPUT_DIR);
    std::string uv_file_name  = "";
    uint32_t    device_id     = 0;
    double      learning_rate = 1e-9;
    uint32_t    num_iter      = 100;
    char**      argv;
    int         argc;
} Arg;


using namespace rxmesh;

template <typename T, typename ProblemT>
void parameterize(RXMeshStatic& rx, ProblemT& problem)
{
    auto coordinates = *rx.get_input_vertex_coordinates();
        
    auto rest_shape =
        *rx.add_face_attribute<Eigen::Matrix<T, 2, 2>>("fRestShape", 1);

    if (Arg.uv_file_name.empty()) {
        tutte_embedding(rx, coordinates, *problem.opt_var);
    } else {
        std::vector<std::vector<uint32_t>> fv;
        std::vector<std::vector<float>>    uv;
        import_obj(Arg.uv_file_name, uv, fv);
        if (uv.size() != rx.get_num_vertices()) {
            RXMESH_ERROR(
                "Number of vertices in the the input UV file {} does not match "
                "the number of vertices in the mesh {}.",
                uv.size(),
                rx.get_num_vertices());
        }
        rx.for_each_vertex(HOST, [&](const VertexHandle vh) {
            uint32_t id = rx.map_to_global(vh);

            (*problem.opt_var)(vh, 0) = uv[id][0];
            (*problem.opt_var)(vh, 1) = uv[id][1];
        });

        problem.opt_var->move(HOST, DEVICE);
    }

#if USE_POLYSCOPE
    rx.get_polyscope_mesh()->addVertexParameterizationQuantity(
        "uv_tutte", *problem.opt_var);
#endif

    constexpr uint32_t blockThreads = 256;

    // 1) compute rest shape
    rx.for_each<Op::FV, blockThreads>(
        [=] __device__(const FaceHandle& fh, const VertexIterator& iter) {
            const VertexHandle v0 = iter[0];
            const VertexHandle v1 = iter[1];
            const VertexHandle v2 = iter[2];

            assert(v0.is_valid() && v1.is_valid() && v2.is_valid());

            // 3d position
            Eigen::Vector3<T> ar_3d = coordinates.to_eigen<3>(v0);
            Eigen::Vector3<T> br_3d = coordinates.to_eigen<3>(v1);
            Eigen::Vector3<T> cr_3d = coordinates.to_eigen<3>(v2);

            // Local 2D coordinate system
            Eigen::Vector3<T> n  = (br_3d - ar_3d).cross(cr_3d - ar_3d);
            Eigen::Vector3<T> b1 = (br_3d - ar_3d).normalized();
            Eigen::Vector3<T> b2 = n.cross(b1).normalized();

            // Express a, b, c in local 2D coordinates system
            Eigen::Vector2<T> ar_2d(T(0.0), T(0.0));
            Eigen::Vector2<T> br_2d((br_3d - ar_3d).dot(b1), T(0.0));
            Eigen::Vector2<T> cr_2d((cr_3d - ar_3d).dot(b1),
                                    (cr_3d - ar_3d).dot(b2));

            // Save 2-by-2 matrix with edge vectors as columns
            Eigen::Matrix<T, 2, 2> fout = col_mat(br_2d - ar_2d, cr_2d - ar_2d);

            rest_shape(fh) = fout;
        });


    // add energy term
    problem.template add_term<Op::FV>(
        [=] __device__(const auto& fh, const auto& iter, auto& opt_var) {
            // fh is a face handle
            // iter is an iterator over fh's vertices
            // opt_var is the uv coordinates

            assert(iter[0].is_valid() && iter[1].is_valid() &&
                   iter[2].is_valid());

            assert(iter.size() == 3);

            using ActiveT = ACTIVE_TYPE(fh);

            // uv
            Eigen::Vector2<ActiveT> a = opt_var.template active<2>(fh, iter, 0);
            Eigen::Vector2<ActiveT> b = opt_var.template active<2>(fh, iter, 1);
            Eigen::Vector2<ActiveT> c = opt_var.template active<2>(fh, iter, 2);


            // Triangle flipped?
            Eigen::Matrix<ActiveT, 2, 2> M = col_mat(b - a, c - a);

            if (M.determinant() <= 0.0) {
                using PassiveT = PassiveType<ActiveT>;
                return ActiveT(std::numeric_limits<PassiveT>::max());
            }

            // Get constant 2D rest shape and area of triangle t
            const Eigen::Matrix<T, 2, 2> Mr = rest_shape(fh);

            const T A = T(0.5) * Mr.determinant();

            // Compute symmetric Dirichlet energy
            Eigen::Matrix<ActiveT, 2, 2> J = M * Mr.inverse();

            ActiveT res = A * (J.squaredNorm() + J.inverse().squaredNorm());


            return res;
        });


    GradientDescent gd(problem, Arg.learning_rate);

    GPUTimer timer;
    timer.start();

    problem.eval_terms();
    const T initial_energy = problem.get_current_loss();
    for (uint32_t iter = 0; iter < Arg.num_iter; ++iter) {
        gd.take_step();
        if (iter + 1 < Arg.num_iter) {
            problem.eval_terms();
        }
    }
    problem.eval_terms_passive();
    const T final_energy = problem.get_current_loss();

    timer.stop();
    RXMESH_INFO(
        "Parametrization GD: iterations= {}, energy= {} -> {}, time= {} (ms)",
        Arg.num_iter,
        initial_energy,
        final_energy,
        timer.elapsed_millis());

    problem.opt_var->move(DEVICE, HOST);

#if USE_POLYSCOPE
    rx.get_polyscope_mesh()->addVertexParameterizationQuantity(
        "uv_opt", *problem.opt_var);
    polyscope::show();
#endif
}

int main(int argc, char** argv)
{
    using T = float;

    CLI::App app{
        "Param - Mesh parametrization using symmetric Dirichlet energy"};

    app.add_option("-i,--input", Arg.obj_file_name, "Input OBJ mesh file")
        ->default_val(std::string(STRINGIFY(INPUT_DIR) "bunnyhead.obj"));

    app.add_option("--uv",
                   Arg.uv_file_name,
                   "Input UV OBJ file (if empty, will compute tutte embedding)")
        ->default_val(std::string(""));

    app.add_option("-o,--output", Arg.output_folder, "JSON file output folder")
        ->default_val(std::string(STRINGIFY(OUTPUT_DIR)));

    app.add_option("--lr", Arg.learning_rate, "Gradient descent learning rate")
        ->default_val(1e-9);
    app.add_option(
           "-n,--iter", Arg.num_iter, "Number of gradient descent iterations")
        ->default_val(100u);

    app.add_option("-d,--device_id", Arg.device_id, "GPU device ID")
        ->default_val(0u);

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        return app.exit(e);
    }

    rx_init(Arg.device_id);

    Arg.argv = argv;
    Arg.argc = argc;

    RXMESH_INFO("input= {}", Arg.obj_file_name);
    RXMESH_INFO("uv_file= {}", Arg.uv_file_name);
    RXMESH_INFO("output_folder= {}", Arg.output_folder);
    RXMESH_INFO("lr= {}", Arg.learning_rate);
    RXMESH_INFO("iter= {}", Arg.num_iter);
    RXMESH_INFO("device_id= {}", Arg.device_id);

    RXMeshStatic rx(Arg.obj_file_name);

    if (rx.is_closed()) {
        RXMESH_ERROR(
            "The input mesh is closed. The input mesh should have boundaries.");
        return EXIT_FAILURE;
    }

    constexpr int VariableDim = 2;

    using ProblemT = DiffScalarProblem<T, VariableDim, VertexHandle, false>;
    ProblemT problem(rx, false);
    parameterize<T>(rx, problem);

    return 0;
}