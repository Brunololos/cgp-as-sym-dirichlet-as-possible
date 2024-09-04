#include "my_svd.hpp"

std::tuple<Eigen::MatrixX<ADDouble9>, Eigen::MatrixX<ADDouble9>, Eigen::MatrixX<ADDouble9>> my_svd(Eigen::MatrixX<ADDouble9> M)
{
    // Eigen::MatrixX<ADDouble9> MMT = M * M.transpose();
    Eigen::MatrixX<ADDouble9> MTM = M.transpose() * M;

/*     Eigen::MatrixX<ADDouble9> U1;
    Eigen::MatrixX<ADDouble9> V1; */

    Eigen::VectorX<ADDouble9> eigenvalues;
    Eigen::MatrixX<ADDouble9> eigenvectors;

	std::pair<Eigen::VectorX<ADDouble9>, Eigen::MatrixX<ADDouble9>> EVD = my_evd(MTM);
    eigenvalues = EVD.first;
    eigenvectors = EVD.second;

    // printTinyAD9Matrix("MTM", MTM);

    // std::cout << "eigen values/vectors:" << std::endl;
    // printTinyAD9Matrix("eigen values", eigenvalues);
    // printTinyAD9Matrix("eigen vectors", eigenvectors);

    // compute singular values & right singular vectors
    Eigen::VectorX<ADDouble9> singularvalues = eigenvalues;
    Eigen::MatrixX<ADDouble9> singularvectors = eigenvectors/* .colwise().normalized() */;
    for (int i = 0; i < singularvalues.size(); i++)
    {
        singularvalues(i) = sqrt(abs(singularvalues(i)));
    }

    // std::cout << BLUE;
    // std::cout << "calculated singular values/vectors:" << std::endl;
    // printTinyAD9Matrix("singular values", singularvalues);
    // printTinyAD9Matrix("singular vectors", singularvectors);
    // std::cout << RESET;


    // TODO: replace eigenvalues / eigenvectors with singular counterparts
    Eigen::MatrixX<ADDouble9> V = singularvectors;

    Eigen::MatrixX<ADDouble9> S = singularvalues.asDiagonal();

    Eigen::MatrixX<ADDouble9> U = (M * V * diagonal_inverse(S));

    // NOTE: for the zero singular values we complete the orthonormal basis (This right now works only in 3d)
    // TODO: should better use Gram-Schmidt
    for (int i = 0; i < singularvalues.size(); i++)
    {
        // hacky way, that only works for 3d
        if (singularvalues(i) == 0)
        {
            if (i == 0)
            {
                U = Eigen::MatrixX<ADDouble9>::Identity(U.rows(), U.cols());
                break;
            }
            else if (i == 1)
            {
                U(0, i) = - U(1, 0);
                U(1, i) = U(0, 0);
                U(2, i) = U(2, 0);
            }
            else if (i == 2)
            {
                Eigen::Vector3<ADDouble9> v0(U(0, 0), U(1, 0), U(2, 0));
                Eigen::Vector3<ADDouble9> v1(U(0, 1), U(1, 1), U(2, 1));
                Eigen::Vector3<ADDouble9> n = v0.cross(v1).normalized();
                U(0, i) = n(0);
                U(1, i) = n(1);
                U(2, i) = n(2);
            }
        }
    }
    return std::tuple<Eigen::MatrixX<ADDouble9>, Eigen::MatrixX<ADDouble9>, Eigen::MatrixX<ADDouble9>>(U, S, V);
}

//std::tuple<Eigen::MatrixX<ADDouble9>, Eigen::VectorX<ADDouble9>, Eigen::MatrixX<ADDouble9>> my_evd(Eigen::MatrixX<ADDouble9> M, Eigen::VectorX<ADDouble9> eigenvalues = Eigen::VectorX<ADDouble9>::Zero(3), Eigen::MatrixX<ADDouble9> eigenvectors = Eigen::MatrixX<ADDouble9>::Zero(3, 3), int eig_count = 0)
std::pair<Eigen::VectorX<ADDouble9>, Eigen::MatrixX<ADDouble9>> my_evd(Eigen::MatrixX<ADDouble9> M, Eigen::VectorX<ADDouble9> eigenvalues, Eigen::MatrixX<ADDouble9> eigenvectors, int eig_count)
{
    // std::cout << YELLOW;
    // std::cout << "Beginning eigen decomposition of matrix: MTM" << std::endl;
    // printTinyAD9Matrix("M", M);
    // std::cout << RESET;
    // NOTE: eig_count & number of eigenvalues & eigenvectors passed, need to be equal to be congruent && eigenvalues & M have to have the same length
    int m_size = M.rows();
    Eigen::VectorX<ADDouble9> vec = Eigen::VectorX<ADDouble9>::Ones(m_size);
    bool is_origin_call = false;

    Eigen::MatrixX<ADDouble9> Mi;
    if (eigenvalues.size() == 0 && eigenvectors.size() == 0)
    {
        // // std::cout << repeat("--", eig_count) + "> ";
        // // std::cout << "set eigenvalues, eigenvectors to sizes (" << m_size << "), (" << m_size << ", " << m_size << ") and Mi to M with size (" << M.rows() << ", " << M.cols() << ") => ";
        eigenvalues = Eigen::VectorX<ADDouble9>::Zero(m_size);
        eigenvectors = Eigen::MatrixX<ADDouble9>::Zero(m_size, m_size);
        Mi = M;
        is_origin_call = true;
        // // std::cout << "SUCCESS" << std::endl;
    }

    Eigen::MatrixX<ADDouble9> m(m_size, 100);
    ADDouble9 lambda_old = 0;

    int index = 0;
    bool is_eval = false;

    // // // std::cout << repeat("--", eig_count) + "> ";
    // // // std::cout << "entering is_eval loop" << std::endl;
    // find maximum eigenvalue
    while(!is_eval)
    {
        // //// std::cout << "entering is_eval iteration: " << index << std::endl;
        // extend m horizontally by 100 columns
        if (index % 100 == 0)
        {
            // // // std::cout << repeat("--", eig_count) + "> ";
            // // // std::cout << "extending m with size: (" << m.rows() << ", " << m.cols() << ") by 100 columns => ";
            Eigen::MatrixX<ADDouble9> D(m.rows(), m.cols() + 100);
            D << m, Eigen::MatrixX<ADDouble9>::Zero(m.rows(), 100);
            m = D;

            // // // std::cout << "SUCCESS" << std::endl;
        }

        for (int row = 0; row < m_size; row++)
        {
            m(row, index) = 0;
            for (int col = 0; col < m_size; col++)
            {
                m(row, index) += M(row, col) * vec(col);
            }
        }

        for (int col = 0; col < m_size; col++)
        {
            vec(col) = m(col, index);
        }

        if (index > 0)
        {
            ADDouble9 lambda = (m(0, index - 1) != 0) ? (m(0, index) / m(0, index - 1)) : m(0, index);
            is_eval = (abs(lambda - lambda_old) < /* 0.000000000000001 */ 0.0000000001) ? true : false;

            lambda = (abs(lambda) >= /* 0.000000000000001 */ 0.000001) ? lambda : 0;
            eigenvalues(eig_count) = lambda;
            lambda_old = lambda;
        }

        index++;
    }

    // std::cout << repeat("--", eig_count) + "> ";
    // std::cout << dye("calculated eigenvalue: " + std::to_string(eigenvalues(eig_count).val), YELLOW) << std::endl;

    Eigen::MatrixX<ADDouble9> Mn;

    if (m_size > 1)
    {
        // // std::cout << repeat("--", eig_count) + "> ";
        // // std::cout << "entered m_size > 1" << std::endl;
        Eigen::MatrixX<ADDouble9> matrix_target = Eigen::MatrixX<ADDouble9>::Zero(m_size, m_size);

        for (int row = 0; row < m_size; row++)
        {
            for (int col = 0; col < m_size; col++)
            {
                // TODO: here was a weird backslash
                matrix_target(row, col) = (row == col) ? /* \ */ (M(row, col) - eigenvalues(eig_count)) : M(row, col);
            }
        }
        // // // std::cout << repeat("--", eig_count) + "> ";
        // // // std::cout << "initialised matrix_target: " << std::endl;
        // // // printTinyAD9Matrix("matrix_target", matrix_target);

        Eigen::VectorX<ADDouble9> eigenvector;
        std::pair<Eigen::MatrixX<ADDouble9>, Eigen::VectorX<ADDouble9>> JGT = my_jordan_gaussian_transform(matrix_target);
        matrix_target = JGT.first;
        eigenvector = JGT.second;

        // // std::cout << YELLOW;
        // // std::cout << repeat("--", eig_count) + "> ";
        // // std::cout << "calculated (jordan gaussian) eigenvector and updated matrix_target: " << std::endl;
        // // printTinyAD9Matrix("eigenvector", eigenvector);
        // // // printTinyAD9Matrix("matrix_target", matrix_target);
        // // std::cout << RESET;

        Eigen::MatrixX<ADDouble9> hermitian = calc_my_hermitian(eigenvector);
        Eigen::MatrixX<ADDouble9> hermitian_inverse = calc_my_hermitian_inverse(eigenvector);

        // // // std::cout << repeat("--", eig_count) + "> ";
        // // // std::cout << "calculated hermitian and hermitian inverse: " << std::endl;
        // // // printTinyAD9Matrix("hermitian", hermitian);
        // // // printTinyAD9Matrix("hermitian_inverse", hermitian_inverse);
        // Eigen::MatrixX<ADDouble9> HA = hermitian * M;
        Eigen::MatrixX<ADDouble9> HiA = hermitian * M * hermitian_inverse;
        Mn = get_my_reduced_matrix(HiA, m_size - 1);

        // // // std::cout << repeat("--", eig_count) + "> ";
        // // // std::cout << "calculated hermitian (M) sandwhich H * M * H_inv: " << std::endl;
        // // // printTinyAD9Matrix("hermitian sandwhich", HiA);

        // // // std::cout << repeat("--", eig_count) + "> ";
        // // // std::cout << "calculated reduced matrix Mn: " << std::endl;
        // // // printTinyAD9Matrix("Mn", Mn);

        // // // std::cout << repeat("--", eig_count) + "> ";
        // // // std::cout << "exiting m_size > 1" << std::endl;
    }

    if (eig_count < eigenvalues.size() - 1)
    {
        // // std::cout << repeat("--", eig_count) + "> ";
        // // std::cout << "recursing with eigencount: " << eig_count + 1 << " and the matrices:" << std::endl;
        // // printTinyAD9Matrix("Mn", Mn);
        // // printTinyAD9Matrix("eigenvalues", eigenvalues);
        std::pair<Eigen::VectorX<ADDouble9>, Eigen::MatrixX<ADDouble9>> EVD = my_evd(Mn, eigenvalues, eigenvectors, eig_count + 1);
        eigenvalues = EVD.first;
    }

    // find eigenvectors
    if (/* m_size <= 1 */ is_origin_call)
    {
        // std::cout << repeat("--", eig_count) + "> ";
        // std::cout << "entered m_size <= 1 with eigenvalues:" << std::endl;
        // printTinyAD9Matrix("eigenvalues", eigenvalues);

        for (int index = 0; index < eigenvalues.size(); index++)
        {
            // std::cout << YELLOW;
            // std::cout << repeat("--", eig_count) + "> ";
            // std::cout << "calculating eigenvector: " << index << " for lambda: " << eigenvalues(index).val << std::endl;
            // std::cout << RESET;
            ADDouble9 lambda = eigenvalues(index);
            Eigen::MatrixX<ADDouble9> matrix_target = Eigen::MatrixX<ADDouble9>::Zero(Mi.rows(), Mi.cols());

            for (int row = 0; row < Mi.rows(); row++)
            {
                for (int col = 0; col < Mi.cols(); col++)
                {
                    matrix_target(row, col) = (row == col) ? (Mi(row, col) - lambda) : Mi(row, col);
                }
            }

            // std::cout << repeat("--", eig_count) + "> ";
            // std::cout << "built matrix_target: " << std::endl;
            // printTinyAD9Matrix("matrix_target", matrix_target);
            
            std::pair<Eigen::MatrixX<ADDouble9>, Eigen::VectorX<ADDouble9>> JGT = my_jordan_gaussian_transform(matrix_target);

            matrix_target = JGT.first;
            eigenvectors.col(index) = JGT.second;

            // std::cout << CYAN;
            // std::cout << repeat("--", eig_count) + "> ";
            // std::cout << "calculated (jordan gaussian) eigenvector and updated matrix_target: " << std::endl;
            // printTinyAD9Matrix("single eigenvector", JGT.second);
            // printTinyAD9Matrix("current eigenvectors", eigenvectors);
            // std::cout << RESET;
            // // printTinyAD9Matrix("matrix_target", matrix_target);
            // normalises eigenvectors => right singular vectors
            ADDouble9 eigsum_sq = 0;
            for (int v = 0; v < eigenvectors.rows(); v++)
            {
                //eigsum_sq += eigenvectors(index, v)*eigenvectors(index, v);
                eigsum_sq += eigenvectors(v, index)*eigenvectors(v, index);
            }
            for (int v = 0; v < eigenvectors.rows(); v++)
            {
                eigenvectors(v, index) = (sqrt(eigsum_sq == 0)) ? eigenvectors(v, index) : eigenvectors(v, index) / sqrt(eigsum_sq);
            }
            // std::cout << "Normalised eigenvector: " << index << std::endl;
            // printTinyAD9Matrix("current eigenvectors", eigenvectors);
            // converts eigenvalues to singular values
            //eigenvalues(index) = sqrt(eigenvalues(index));
        }

        // // std::cout << repeat("--", eig_count) + "> ";
        // // std::cout << "returning with eigen vector" << std::endl;
        // // printTinyAD9Matrix("eigenvectors", eigenvectors);

        return std::pair<Eigen::VectorX<ADDouble9>, Eigen::MatrixX<ADDouble9>>(eigenvalues, eigenvectors);
    }
    //compute evd recursive

    // // // std::cout << repeat("--", eig_count) + "> ";
    // // // std::cout << "recursing with eigencount: " << eig_count + 1 << " and the matrices:" << std::endl;
    // // // printTinyAD9Matrix("Mn", Mn);
    // // // printTinyAD9Matrix("eigenvalues", eigenvalues);
    // //// printTinyAD9Matrix("eigenvectors", eigenvectors);

    // TODO: maybe switch matrix.rows() calls everywhere, as the dude used row to specify the position in a row, not the index of a row.
    //return std::tuple<Eigen::MatrixX<ADDouble9>, Eigen::VectorX<ADDouble9>, Eigen::MatrixX<ADDouble9>>(M, eigenvalues, eigenvectors);
    // return my_evd(Mn, eigenvalues, eigenvectors, eig_count + 1);

    return std::pair<Eigen::VectorX<ADDouble9>, Eigen::MatrixX<ADDouble9>>(eigenvalues, eigenvectors);
}

std::pair<Eigen::MatrixX<ADDouble9>, Eigen::VectorX<ADDouble9>> my_jordan_gaussian_transform(Eigen::MatrixX<ADDouble9> M)
{
    Eigen::VectorX<ADDouble9> eigenvector = Eigen::VectorX<ADDouble9>::Zero(M.rows());
    const double eps = 0.00000001;
    bool eigenv_found = false;
    int eigenv_cidx = 0;

    // std::cout << PURPLE;
    // std::cout << "Solving Matrix M for eigenvector (via jordan gaussian):" << std::endl;
    // printTinyAD9Matrix("M", M);
    // std::cout << RESET;


    // TODO: check s < M.cols() - 1
    for (int s = 0; s < M.cols() - 1 && !eigenv_found; s++)
    {
        int col = s;
        ADDouble9 alpha = M(s, s);
        // Create leading 1 in current row: (divide row by alpha)
        while (col < M.cols() && alpha != 0 && alpha != 1)
        {
            M(s, col++) /= alpha;
        }

        // std::cout << "divided row: " << s << " by alpha = " << alpha.val << std::endl;
        // printTinyAD9Matrix("M", M);
        // if alpha == 0 swap rows i and i+1
        for (int col = s; col < M.cols() && alpha == 0; col++)
        {
            ADDouble9 temp = M(s, col);
            M(s, col) = M(s + 1, col);
            M(s + 1, col) = temp;
        }
        if (alpha == 0)
        {
            // std::cout << YELLOW;
            // std::cout << "swapped rows: " << std::endl;
            // printTinyAD9Matrix("M", M);
            // std::cout << RESET;
        }

        // create zero's belowa and above the current leading 1
        for (int row = 0; row < M.rows(); row++)
        {
            ADDouble9 gamma = M(row, s);
            // // std::cout << " found row: " << row << " gamma: " << gamma.val << std::endl;
            for (int col = s; col < M.cols() && row != s; col++)
            {
                M(row, col) = M(row, col) - M(s, col) * gamma;
            }
            // if (row != s) { std::cout << " subtracted row: " << s << " with factor gamma: " << gamma.val << " from row: " << row << std::endl; }
        }
        // printTinyAD9Matrix("M", M);

        int row = 0;
        // if finished
        while (row < M.rows() && (s == M.rows() - 1/* 2 */ || abs(M(s + 1, s + 1)) < eps))
        {
            eigenvector(row) = -M(row++, s + 1);
            eigenv_cidx++;
            // std::cout << "setting eigvector index: " << row << std::endl;
        }
        if (s == M.rows() - 1 || abs(M(s + 1, s + 1)) < eps)
        {
            // std::cout << GREEN;
            // std::cout << "Calculated JGT & set eigenvector:" << std::endl;
            // printTinyAD9Matrix("jgt preliminary result", eigenvector);
            // std::cout << RESET;
        }
        // if finished, end iteration by eigenv_found & set last entry of eigenvector to 1.
        if (eigenv_cidx == M.rows()/* s == M.cols() - 2 */)
        {
            eigenv_found = true;
            eigenvector(s + 1) = 1;
            
            // std::cout << GREEN;
            // std::cout << "Finalised eigenvector:" << std::endl;
            // printTinyAD9Matrix("jgt result eigenvector", eigenvector);
            // std::cout << RESET;
            // precision truncation
            for (int index = s + 1; index < eigenvector.size(); index++)
            {
                eigenvector(index) = (abs(eigenvector(index)) >= eps) ? eigenvector(index) : 0;
            }
        }

    }
    return std::pair<Eigen::MatrixX<ADDouble9>, Eigen::VectorX<ADDouble9>>(M, eigenvector);
}

// TODO: this could divide by zero! And it does produce NAN's!!!
Eigen::MatrixX<ADDouble9> calc_my_hermitian(Eigen::VectorX<ADDouble9> eigenvector)
{
    if (eigenvector(0) == 0.0) {
        // // std::cout << RED << "ENCOUNTERED zero first eigenvector value in calc hermitian" << RESET << std::endl;
    }
    int eig_size = eigenvector.size();
    Eigen::MatrixX<ADDouble9> H = Eigen::MatrixX<ADDouble9>::Zero(eig_size, eig_size);

    H(0, 0) = (eigenvector(0) != 0) ? 1 / eigenvector(0) : 1;
    for (int row = 1; row < eig_size; row++)
    {
        H(row, 0) = (eigenvector(0) != 0) ? -eigenvector(row) / eigenvector(0) : -eigenvector(row);
        H(row, row) = 1;
    }
/*     for (int row = 1; row < eigenvector.size(); row++)
    {
        H(row, row) = 1;
    } */

    return H;
}
Eigen::MatrixX<ADDouble9> calc_my_hermitian_inverse(Eigen::VectorX<ADDouble9> eigenvector)
{
    int eig_size = eigenvector.size();
    Eigen::MatrixX<ADDouble9> Hi = Eigen::MatrixX<ADDouble9>::Zero(eig_size, eig_size);

    Hi(0, 0) = eigenvector(0);
    for (int row = 1; row < eig_size; row++)
    {
        Hi(row, 0) = -eigenvector(row);
        Hi(row, row) = 1;
    }
/*     for (int row = 1; row < eigenvector.size(); row++)
    {
        Hi(row, row) = 1;
    } */

    return Hi;
}

// This is just a slicing out of the bottom right corner square matrix of size new_size
Eigen::MatrixX<ADDouble9> get_my_reduced_matrix(Eigen::MatrixX<ADDouble9> M, int new_size)
{
    Eigen::MatrixX<ADDouble9> new_matrix = Eigen::MatrixX<ADDouble9>::Zero(new_size, new_size);
    int index = M.rows() - new_size;
    int row = index;
    int row_n = 0;
    while (row < M.rows())
    {
        int col = index;
        int col_n = 0;
        while (col < M.rows())
        {
            new_matrix(row_n, col_n++) = M(row, col++);
        }
        row++;
        row_n++;
    }

    return new_matrix;
}

Eigen::MatrixX<ADDouble9> diagonal_inverse(Eigen::MatrixX<ADDouble9> M)
{
    Eigen::MatrixX<ADDouble9> M_inv = Eigen::MatrixX<ADDouble9>::Zero(M.rows(), M.cols());
	for (int index = 0; index < M.cols(); index++)
	{
        if (M(index, index) == 0.0) {
            // // std::cout << RED << "ENCOUNTERED zero M entry in diagonal inverse" << RESET << std::endl;
        }
        M_inv(index, index) = (M(index, index) != 0) ? 1.0 / M(index, index) : 0.0; // TODO: 0.0 or 1.0 ?
	}
    return M_inv;
}

// MARK: SVD for doubles
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> my_svdd(Eigen::MatrixXd M)
{
    Eigen::MatrixXd MTM = M.transpose() * M;

    Eigen::VectorXd eigenvalues;
    Eigen::MatrixXd eigenvectors;

	std::pair<Eigen::VectorXd, Eigen::MatrixXd> EVD = my_evdd(MTM);
    eigenvalues = EVD.first;
    eigenvectors = EVD.second;

    // compute singular values & right singular vectors
    Eigen::VectorXd singularvalues = eigenvalues;
    Eigen::MatrixXd singularvectors = eigenvectors/* .colwise().normalized() */;
    for (int i = 0; i < singularvalues.size(); i++)
    {
        singularvalues(i) = sqrt(abs(singularvalues(i)));
    }

    Eigen::MatrixXd V = singularvectors;
    Eigen::MatrixXd S = singularvalues.asDiagonal();
    Eigen::MatrixXd U = (M * V * diagonal_inversed(S));
    // NOTE: for the zero singular values we complete the orthonormal basis (This right now works only in 3d)
    // TODO: should better use Gram-Schmidt
    for (int i = 0; i < singularvalues.size(); i++)
    {
        // hacky way, that only works for 3d
        if (singularvalues(i) == 0)
        {
            if (i == 0)
            {
                U = Eigen::MatrixXd::Identity(U.rows(), U.cols());
                break;
            }
            else if (i == 1)
            {
                U(0, i) = - U(1, 0);
                U(1, i) = U(0, 0);
                U(2, i) = U(2, 0);
            }
            else if (i == 2)
            {
                Eigen::Vector3d v0(U(0, 0), U(1, 0), U(2, 0));
                Eigen::Vector3d v1(U(0, 1), U(1, 1), U(2, 1));
                Eigen::Vector3d n = v0.cross(v1).normalized();
                U(0, i) = n(0);
                U(1, i) = n(1);
                U(2, i) = n(2);
            }
        }
    }
    return std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>(U, S, V);
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> my_evdd(Eigen::MatrixXd M, Eigen::VectorXd eigenvalues, Eigen::MatrixXd eigenvectors, int eig_count)
{
    // NOTE: eig_count & number of eigenvalues & eigenvectors passed, need to be equal to be congruent && eigenvalues & M have to have the same length
    int m_size = M.rows();
    Eigen::VectorXd vec = Eigen::VectorXd::Ones(m_size);
    bool is_origin_call = false;

    Eigen::MatrixXd Mi;
    if (eigenvalues.size() == 0 && eigenvectors.size() == 0)
    {
        eigenvalues = Eigen::VectorXd::Zero(m_size);
        eigenvectors = Eigen::MatrixXd::Zero(m_size, m_size);
        Mi = M;
        is_origin_call = true;
    }

    Eigen::MatrixXd m(m_size, 100);
    double lambda_old = 0;

    int index = 0;
    bool is_eval = false;

    // find maximum eigenvalue
    while(!is_eval)
    {
        // extend m horizontally by 100 columns
        if (index % 100 == 0)
        {
            Eigen::MatrixXd D(m.rows(), m.cols() + 100);
            D << m, Eigen::MatrixXd::Zero(m.rows(), 100);
            m = D;

        }

        for (int row = 0; row < m_size; row++)
        {
            m(row, index) = 0;
            for (int col = 0; col < m_size; col++)
            {
                m(row, index) += M(row, col) * vec(col);
            }
        }

        for (int col = 0; col < m_size; col++)
        {
            vec(col) = m(col, index);
        }

        if (index > 0)
        {
            // TODO: there was a weird backslash here
            double lambda = (m(0, index - 1) != 0) ? /* \ */ (m(0, index) / m(0, index - 1)) : m(0, index);
            is_eval = (abs(lambda - lambda_old) < /* 0.000000000000001 */0.0000000001) ? true : false;

            lambda = (abs(lambda) >= /* 0.000000000000001 */0.000001) ? lambda : 0;
            eigenvalues(eig_count) = lambda;
            lambda_old = lambda;
        }

        index++;
    }


    Eigen::MatrixXd Mn;

    if (m_size > 1)
    {
        Eigen::MatrixXd matrix_target = Eigen::MatrixXd::Zero(m_size, m_size);

        for (int row = 0; row < m_size; row++)
        {
            for (int col = 0; col < m_size; col++)
            {
                // TODO: here was a weird backslash
                matrix_target(row, col) = (row == col) ? /* \ */ (M(row, col) - eigenvalues(eig_count)) : M(row, col);
            }
        }

        Eigen::VectorXd eigenvector;
        std::pair<Eigen::MatrixXd, Eigen::VectorXd> JGT = my_jordan_gaussian_transformd(matrix_target);
        matrix_target = JGT.first;
        eigenvector = JGT.second;

        Eigen::MatrixXd hermitian = calc_my_hermitiand(eigenvector);
        Eigen::MatrixXd hermitian_inverse = calc_my_hermitian_inversed(eigenvector);

        Eigen::MatrixXd HiA = hermitian * M * hermitian_inverse;
        Mn = get_my_reduced_matrixd(HiA, m_size - 1);

    }

    if (eig_count < eigenvalues.size() - 1)
    {
        std::pair<Eigen::VectorXd, Eigen::MatrixXd> EVD = my_evdd(Mn, eigenvalues, eigenvectors, eig_count + 1);
        eigenvalues = EVD.first;
    }

    if (/* m_size <= 1 */ is_origin_call)
    {
        for (int index = 0; index < eigenvalues.size(); index++)
        {
            double lambda = eigenvalues(index);
            Eigen::MatrixXd matrix_target = Eigen::MatrixXd::Zero(Mi.rows(), Mi.cols());

            for (int row = 0; row < Mi.rows(); row++)
            {
                for (int col = 0; col < Mi.cols(); col++)
                {
                    // TODO: there was a weird backslash here
                    matrix_target(row, col) = (row == col) ? /* \ */ (Mi(row, col) - lambda) : Mi(row, col);
                }
            }

            std::pair<Eigen::MatrixXd, Eigen::VectorXd> JGT = my_jordan_gaussian_transformd(matrix_target);

            matrix_target = JGT.first;

            //Eigen::MatrixXd new_eigenvectors(eigenvectors.rows(), eigenvectors.cols() + 1);
            //new_eigenvectors << eigenvectors, JGT.second;
            //eigenvectors = new_eigenvectors;
            eigenvectors.col(index) = JGT.second;

            // normalises eigenvectors => right singular vectors
            double eigsum_sq = 0;
            for (int v = 0; v < eigenvectors.rows(); v++)
            {
                //eigsum_sq += eigenvectors(index, v)*eigenvectors(index, v);
                eigsum_sq += eigenvectors(v, index)*eigenvectors(v, index);
            }
            for (int v = 0; v < eigenvectors.rows(); v++)
            {
                eigenvectors(v, index) = (sqrt(eigsum_sq == 0)) ? eigenvectors(v, index) : eigenvectors(v, index) / sqrt(eigsum_sq);
            }
            // converts eigenvalues to singular values
            //eigenvalues(index) = sqrt(eigenvalues(index));
        }

        return std::pair<Eigen::VectorXd, Eigen::MatrixXd>(eigenvalues, eigenvectors);
    }
    //compute evd recursive

    // TODO: maybe switch matrix.rows() calls everywhere, as the dude used row to specify the position in a row, not the index of a row.
    //return std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd>(M, eigenvalues, eigenvectors);
    // return my_evd(Mn, eigenvalues, eigenvectors, eig_count + 1);

    return std::pair<Eigen::VectorXd, Eigen::MatrixXd>(eigenvalues, eigenvectors);
}

std::pair<Eigen::MatrixXd, Eigen::VectorXd> my_jordan_gaussian_transformd(Eigen::MatrixXd M)
{
    Eigen::VectorXd eigenvector = Eigen::VectorXd::Zero(M.rows());
    const double eps = 0.00000001;
    bool eigenv_found = false;
    int eigenv_cidx = 0;

    for (int s = 0; s < M.cols() - 1 && !eigenv_found; s++)
    {
        int col = s;
        double alpha = M(s, s);
        // Create leading 1 in current row: (divide row by alpha)
        while (col < M.cols() && alpha != 0 && alpha != 1)
        {
            M(s, col++) /= alpha;
        }

        // if alpha == 0 swap rows i and i+1
        for (int col = s; col < M.cols() && alpha == 0; col++)
        {
            double temp = M(s, col);
            M(s, col) = M(s + 1, col);
            M(s + 1, col) = temp;
        }

        // create zero's belowa and above the current leading 1
        for (int row = 0; row < M.rows(); row++)
        {
            double gamma = M(row, s);
            for (int col = s; col < M.cols() && row != s; col++)
            {
                M(row, col) = M(row, col) - M(s, col) * gamma;
            }
        }

        int row = 0;
        // if finished
        while (row < M.rows() && (s == M.rows() - 1/* 2 */ || abs(M(s + 1, s + 1)) < eps))
        {
            eigenvector(row) = -M(row++, s + 1);
            eigenv_cidx++;
        }
        // if finished, end iteration by eigenv_found & set last entry of eigenvector to 1.
        if (eigenv_cidx == M.rows()/* s == M.cols() - 2 */)
        {
            eigenv_found = true;
            eigenvector(s + 1) = 1;

            // precision truncation
            for (int index = s + 1; index < eigenvector.size(); index++)
            {
                eigenvector(index) = (abs(eigenvector(index)) >= eps) ? eigenvector(index) : 0;
            }
        }

    }
    return std::pair<Eigen::MatrixXd, Eigen::VectorXd>(M, eigenvector);
}

// TODO: this could divide by zero! And it does produce NAN's!!!
Eigen::MatrixXd calc_my_hermitiand(Eigen::VectorXd eigenvector)
{
    int eig_size = eigenvector.size();
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(eig_size, eig_size);

    H(0, 0) = (eigenvector(0) != 0) ? 1 / eigenvector(0) : 1;
    for (int row = 1; row < eig_size; row++)
    {
        H(row, 0) = (eigenvector(0) != 0) ? -eigenvector(row) / eigenvector(0) : -eigenvector(row);
        H(row, row) = 1;
    }
/*     for (int row = 1; row < eigenvector.size(); row++)
    {
        H(row, row) = 1;
    } */

    return H;
}
Eigen::MatrixXd calc_my_hermitian_inversed(Eigen::VectorXd eigenvector)
{
    int eig_size = eigenvector.size();
    Eigen::MatrixXd Hi = Eigen::MatrixXd::Zero(eig_size, eig_size);

    Hi(0, 0) = eigenvector(0);
    for (int row = 1; row < eig_size; row++)
    {
        Hi(row, 0) = -eigenvector(row);
        Hi(row, row) = 1;
    }
/*     for (int row = 1; row < eigenvector.size(); row++)
    {
        Hi(row, row) = 1;
    } */

    return Hi;
}

// This is just a slicing out of the bottom right corner square matrix of size new_size
Eigen::MatrixXd get_my_reduced_matrixd(Eigen::MatrixXd M, int new_size)
{
    Eigen::MatrixXd new_matrix = Eigen::MatrixXd::Zero(new_size, new_size);
    int index = M.rows() - new_size;
    int row = index;
    int row_n = 0;
    while (row < M.rows())
    {
        int col = index;
        int col_n = 0;
        while (col < M.rows())
        {
            new_matrix(row_n, col_n++) = M(row, col++);
        }
        row++;
        row_n++;
    }

    return new_matrix;
}

Eigen::MatrixXd diagonal_inversed(Eigen::MatrixXd M)
{
    Eigen::MatrixXd M_inv = Eigen::MatrixXd::Zero(M.rows(), M.cols());
	for (int index = 0; index < M.cols(); index++)
	{
        M_inv(index, index) = (M(index, index) != 0) ? 1.0 / M(index, index) : 1.0; // TODO: 0.0 or 1.0 ?
	}
    return M_inv;
}
