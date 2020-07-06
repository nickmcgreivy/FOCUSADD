import jax.numpy as np
from jax import jit
import numpy as numpy
from jax.ops import index, index_add
import math as m
import tables as tb
from functools import partial
from jax.config import config

config.update("jax_enable_x64", True)

PI = m.pi


class CoilSet:

    """
	CoilSet is a class which represents all of the coils surrounding a plasma surface. The coils
	are represented by two fourier series, one for the coil winding pack centroid and one for the 
	rotation of the coils. 
	"""

    def get_initial_data(surface, input_file=None, args_dict=None):
        """
		There are two methods of initializing the coil set. Both require a surface which the coils surround. 

		The first method of initialization involves an HDF5 file which stores the coil data and metadata. We can supply
		input_file and this tells us a path to the coil data. 

		The second method of initialization involves reading the coils in from args_dict, which is a dictionary
		of the coil metadata. From this metadata and the surface we can initialize the coils around the surface. 
		"""
        if input_file is not None:
            with tb.open_file(input_file, "r") as f:
                (NC, NS, NF, NFR, ln, lb, NNR, NBR, rc, NR,) = f.root.metadata[0]
                fc = np.asarray(f.root.coilSeries[:, :, :])
                fr = np.asarray(f.root.rotationSeries[:, :, :])
        elif args_dict is not None:
            # INITIALIZE COILS TO DEFAULT VALUES
            NC = args_dict["numCoils"]
            NS = args_dict["numSegments"]
            NF = args_dict["numFourierCoils"]
            NFR = args_dict["numFourierRotate"]
            ln = args_dict["lengthNormal"]
            lb = args_dict["lengthBinormal"]
            NNR = args_dict["numNormalRotate"]
            NBR = args_dict["numBinormalRotate"]
            rc = args_dict["radiusCoil"]
            NR = args_dict["numRotate"]
            r_centroid = surface.calc_r_coils(NC, NS, rc)  # Position of centroid
            fc = CoilSet.compute_coil_fourierSeries(NC, NS, NF, r_centroid)
            fr = np.zeros((2, NC, NFR))
        else:
            raise Exception("No file or args_dict passed to initialize coil set. ")

        coil_data = NC, NS, NF, NFR, ln, lb, NNR, NBR, rc, NR
        params = fc, fr
        return coil_data, params

    def compute_coil_fourierSeries(NC, NS, NF, r_centroid):
        """ 
		Takes a set of centroid positions and gives the coefficients
		of the coil fourier series in a single array 

		Inputs:
		r_centroid (nparray): vector of length NC x NS + 1 x 3, initial coil centroid

		Returns:
		6 x NC x NF array with the Fourier Coefficients of the initial coils
		"""
        x = r_centroid[:, 0:-1, 0]  # NC x NS
        y = r_centroid[:, 0:-1, 1]
        z = r_centroid[:, 0:-1, 2]
        xc = np.zeros((NC, NF))
        yc = np.zeros((NC, NF))
        zc = np.zeros((NC, NF))
        xs = np.zeros((NC, NF))
        ys = np.zeros((NC, NF))
        zs = np.zeros((NC, NF))
        xc = index_add(xc, index[:, 0], np.sum(x, axis=1) / NS)
        yc = index_add(yc, index[:, 0], np.sum(y, axis=1) / NS)
        zc = index_add(zc, index[:, 0], np.sum(z, axis=1) / NS)
        theta = np.linspace(0, 2 * PI, NS + 1)[0:NS]
        for m in range(1, NF):
            xc = index_add(
                xc, index[:, m], 2.0 * np.sum(x * np.cos(m * theta), axis=1) / NS
            )
            yc = index_add(
                yc, index[:, m], 2.0 * np.sum(y * np.cos(m * theta), axis=1) / NS
            )
            zc = index_add(
                zc, index[:, m], 2.0 * np.sum(z * np.cos(m * theta), axis=1) / NS
            )
            xs = index_add(
                xs, index[:, m], 2.0 * np.sum(x * np.sin(m * theta), axis=1) / NS
            )
            ys = index_add(
                ys, index[:, m], 2.0 * np.sum(y * np.sin(m * theta), axis=1) / NS
            )
            zs = index_add(
                zs, index[:, m], 2.0 * np.sum(z * np.sin(m * theta), axis=1) / NS
            )
        return np.asarray([xc, yc, zc, xs, ys, zs])  # 6 x NC x NF

    def write(coil_data, params, output_file):
        """ Write coils in HDF5 output format.
		Input:

		output_file (string): Path to outputfile, string should include .hdf5 format


		"""
        NC, NS, NF, NFR, ln, lb, NNR, NBR, rc, NR = coil_data
        fc, fr = params
        with tb.open_file(output_file, "w") as f:
            metadata = numpy.dtype(
                [
                    ("NC", int),
                    ("NS", int),
                    ("NF", int),
                    ("NFR", int),
                    ("ln", float),
                    ("lb", float),
                    ("NNR", int),
                    ("NBR", int),
                    ("rc", float),
                    ("NR", int),
                ]
            )
            arr = numpy.array(
                [(NC, NS, NF, NFR, ln, lb, NNR, NBR, rc, NR,)], dtype=metadata,
            )
            f.create_table("/", "metadata", metadata)
            f.root.metadata.append(arr)
            f.create_array("/", "coilSeries", numpy.asarray(fc))
            f.create_array("/", "rotationSeries", numpy.asarray(fr))

    @partial(jit, static_argnums=(0,))
    def get_outputs(coil_data, params):
        """ 
		Takes a tuple of coil parameters and sets the parameters. When the 
		parameters are reset, we need to update the other variables like the coil position, frenet frame, etc. 

		Inputs: 
		params (tuple of numpy arrays): Tuple of parameters, first array is 6 x NC x NF

		"""
        NC, NS, NF, NFR, ln, lb, NNR, NBR, rc, NR = coil_data
        theta = np.linspace(0, 2 * PI, NS + 1)
        fc, fr = params
        I = np.ones(NC) / (NNR * NBR)
        # COMPUTE COIL VARIABLES
        r_centroid = CoilSet.compute_r_centroid(coil_data, fc, theta)
        r1 = CoilSet.compute_x1y1z1(coil_data, fc, theta)
        r2 = CoilSet.compute_x2y2z2(coil_data, fc, theta)
        r3 = CoilSet.compute_x3y3z3(coil_data, fc, theta)
        torsion = CoilSet.compute_torsion(r1, r2, r3)
        mean_torsion = CoilSet.compute_mean_torsion(torsion)
        tangent, normal, binormal = CoilSet.compute_com(r1, fc, r_centroid)
        r = CoilSet.compute_r(coil_data, theta, fr, normal, binormal, r_centroid)
        frame = tangent, normal, binormal
        dl = CoilSet.compute_dl(coil_data, theta, params, frame, r1, r2, r_centroid)
        av_length = CoilSet.compute_average_length(r_centroid, NC)
        return I, dl, r, av_length

    def get_frame(coil_data, params):
        NC, NS, NF, NFR, ln, lb, NNR, NBR, rc, NR = coil_data
        theta = np.linspace(0, 2 * PI, NS + 1)
        fc, fr = params
        I = np.ones(NC) / (NNR * NBR)
        # COMPUTE COIL VARIABLES
        r_centroid = CoilSet.compute_r_centroid(coil_data, fc, theta)
        r1 = CoilSet.compute_x1y1z1(coil_data, fc, theta)
        r2 = CoilSet.compute_x2y2z2(coil_data, fc, theta)
        r3 = CoilSet.compute_x3y3z3(coil_data, fc, theta)
        torsion = CoilSet.compute_torsion(r1, r2, r3)
        mean_torsion = CoilSet.compute_mean_torsion(torsion)
        tangent, normal, binormal = CoilSet.compute_com(r1, fc, r_centroid)
        return tangent, normal, binormal

    def get_r_centroid(coil_data, params):
        NC, NS, NF, NFR, ln, lb, NNR, NBR, rc, NR = coil_data
        theta = np.linspace(0, 2 * PI, NS + 1)
        fc, fr = params
        I = np.ones(NC) / (NNR * NBR)
        # COMPUTE COIL VARIABLES
        r_centroid = CoilSet.compute_r_centroid(coil_data, fc, theta)
        return r_centroid

    def unpack_fourier(fc):
        """
		Takes coil fourier series fc and unpacks it into 6 components
		"""
        xc = fc[0]
        yc = fc[1]
        zc = fc[2]
        xs = fc[3]
        ys = fc[4]
        zs = fc[5]
        return xc, yc, zc, xs, ys, zs

    def compute_r_centroid(coil_data, fc, theta):
        NC, NS, NF, _, _, _, _, _, _, _ = coil_data
        """ Computes the position of the winding pack centroid using the coil fourier series """
        xc, yc, zc, xs, ys, zs = CoilSet.unpack_fourier(fc)
        x = np.zeros((NC, NS + 1))
        y = np.zeros((NC, NS + 1))
        z = np.zeros((NC, NS + 1))
        for m in range(NF):
            arg = m * theta
            carg = np.cos(arg)
            sarg = np.sin(arg)
            x += (
                xc[:, np.newaxis, m] * carg[np.newaxis, :]
                + xs[:, np.newaxis, m] * sarg[np.newaxis, :]
            )
            y += (
                yc[:, np.newaxis, m] * carg[np.newaxis, :]
                + ys[:, np.newaxis, m] * sarg[np.newaxis, :]
            )
            z += (
                zc[:, np.newaxis, m] * carg[np.newaxis, :]
                + zs[:, np.newaxis, m] * sarg[np.newaxis, :]
            )
        return np.concatenate(
            (x[:, :, np.newaxis], y[:, :, np.newaxis], z[:, :, np.newaxis]), axis=2
        )

    def compute_x1y1z1(coil_data, fc, theta):
        """ Computes a first derivative of the centroid """
        NC, NS, NF, _, _, _, _, _, _, _ = coil_data
        xc, yc, zc, xs, ys, zs = CoilSet.unpack_fourier(fc)
        x1 = np.zeros((NC, NS + 1))
        y1 = np.zeros((NC, NS + 1))
        z1 = np.zeros((NC, NS + 1))
        for m in range(NF):
            arg = m * theta
            carg = np.cos(arg)
            sarg = np.sin(arg)
            x1 += (
                -m * xc[:, np.newaxis, m] * sarg[np.newaxis, :]
                + m * xs[:, np.newaxis, m] * carg[np.newaxis, :]
            )
            y1 += (
                -m * yc[:, np.newaxis, m] * sarg[np.newaxis, :]
                + m * ys[:, np.newaxis, m] * carg[np.newaxis, :]
            )
            z1 += (
                -m * zc[:, np.newaxis, m] * sarg[np.newaxis, :]
                + m * zs[:, np.newaxis, m] * carg[np.newaxis, :]
            )
        return np.concatenate(
            (x1[:, :, np.newaxis], y1[:, :, np.newaxis], z1[:, :, np.newaxis]), axis=2
        )

    def compute_x2y2z2(coil_data, fc, theta):
        """ Computes a second derivative of the centroid """
        NC, NS, NF, _, _, _, _, _, _, _ = coil_data
        xc, yc, zc, xs, ys, zs = CoilSet.unpack_fourier(fc)
        x2 = np.zeros((NC, NS + 1))
        y2 = np.zeros((NC, NS + 1))
        z2 = np.zeros((NC, NS + 1))
        for m in range(NF):
            m2 = m ** 2
            arg = m * theta
            carg = np.cos(arg)
            sarg = np.sin(arg)
            x2 += (
                -m2 * xc[:, np.newaxis, m] * carg[np.newaxis, :]
                - m2 * xs[:, np.newaxis, m] * sarg[np.newaxis, :]
            )
            y2 += (
                -m2 * yc[:, np.newaxis, m] * carg[np.newaxis, :]
                - m2 * ys[:, np.newaxis, m] * sarg[np.newaxis, :]
            )
            z2 += (
                -m2 * zc[:, np.newaxis, m] * carg[np.newaxis, :]
                - m2 * zs[:, np.newaxis, m] * sarg[np.newaxis, :]
            )
        return np.concatenate(
            (x2[:, :, np.newaxis], y2[:, :, np.newaxis], z2[:, :, np.newaxis]), axis=2
        )

    def compute_x3y3z3(coil_data, fc, theta):
        """ Computes a third derivative of the centroid """
        NC, NS, NF, _, _, _, _, _, _, _ = coil_data
        xc, yc, zc, xs, ys, zs = CoilSet.unpack_fourier(fc)
        x3 = np.zeros((NC, NS + 1))
        y3 = np.zeros((NC, NS + 1))
        z3 = np.zeros((NC, NS + 1))
        for m in range(NF):
            m3 = m ** 3
            arg = m * theta
            carg = np.cos(arg)
            sarg = np.sin(arg)
            x3 += (
                m3 * xc[:, np.newaxis, m] * sarg[np.newaxis, :]
                - m3 * xs[:, np.newaxis, m] * carg[np.newaxis, :]
            )
            y3 += (
                m3 * yc[:, np.newaxis, m] * sarg[np.newaxis, :]
                - m3 * ys[:, np.newaxis, m] * carg[np.newaxis, :]
            )
            z3 += (
                m3 * zc[:, np.newaxis, m] * sarg[np.newaxis, :]
                - m3 * zs[:, np.newaxis, m] * carg[np.newaxis, :]
            )
        return np.concatenate(
            (x3[:, :, np.newaxis], y3[:, :, np.newaxis], z3[:, :, np.newaxis]), axis=2
        )

    def compute_average_length(r_centroid, NC):
        dl_centroid = r_centroid[:, 1:, :] - r_centroid[:, :-1, :]
        return np.sum(np.linalg.norm(dl_centroid, axis=-1)) / (NC)

    def compute_com(r1, fc, r_centroid):
        """ Computes T, N, and B """
        tangent = CoilSet.compute_tangent(r1)
        normal = -CoilSet.compute_normal(fc, r_centroid, tangent)
        binormal = CoilSet.compute_binormal(tangent, normal)
        return tangent, normal, binormal

    def compute_com_deriv(params, frame, r1, r2, r_centroid):
        T, N, B = frame
        fc, _ = params
        tangent_deriv = CoilSet.compute_tangent_deriv(r1, r2)
        normal_deriv = -CoilSet.compute_normal_deriv(
            fc, T, tangent_deriv, r1, r_centroid
        )
        binormal_deriv = CoilSet.compute_binormal_deriv(
            T, N, tangent_deriv, normal_deriv
        )
        return tangent_deriv, normal_deriv, binormal_deriv

    def compute_tangent(r1):
        """
		Computes the tangent vector of the coils. Uses the equation 
		T = dr/d_theta / |dr / d_theta|
		"""
        return r1 / np.linalg.norm(r1, axis=-1)[:, :, np.newaxis]

    def compute_tangent_deriv(r1, r2):
        norm_r1 = np.linalg.norm(r1, axis=-1)
        mag_2 = CoilSet.dot_product_rank3_tensor(r1, r2) / norm_r1 ** 3
        return r2 / norm_r1[:, :, np.newaxis] - r1 * mag_2[:, :, np.newaxis]

    def dot_product_rank3_tensor(a, b):
        return (
            a[:, :, 0] * b[:, :, 0] + a[:, :, 1] * b[:, :, 1] + a[:, :, 2] * b[:, :, 2]
        )

    def compute_coil_com(fc):
        xc, yc, zc, xs, ys, zs = CoilSet.unpack_fourier(fc)  # each of these is NC x NF
        return np.concatenate(
            (xc[:, 0, np.newaxis], yc[:, 0, np.newaxis], zc[:, 0, np.newaxis]), axis=1
        )  # NC x 3

    def compute_normal(fc, r_centroid, tangent):
        r0 = CoilSet.compute_coil_com(fc)
        delta = r_centroid - r0[:, np.newaxis, :]
        dp = CoilSet.dot_product_rank3_tensor(tangent, delta)
        normal = delta - tangent * dp[:, :, np.newaxis]
        mag = np.linalg.norm(normal, axis=-1)
        return normal / mag[:, :, np.newaxis]

    def compute_normal_deriv(fc, T, tangent_deriv, r1, r_centroid):
        r0 = CoilSet.compute_coil_com(fc)
        delta = r_centroid - r0[:, np.newaxis, :]
        dp1 = CoilSet.dot_product_rank3_tensor(T, delta)
        dp2 = CoilSet.dot_product_rank3_tensor(T, r1)
        dp3 = CoilSet.dot_product_rank3_tensor(tangent_deriv, delta)
        numerator = delta - T * dp1[:, :, np.newaxis]
        numerator_norm = np.linalg.norm(numerator, axis=-1)
        numerator_deriv = (
            r1
            - dp1[:, :, np.newaxis] * tangent_deriv
            - T * (dp2 + dp3)[:, :, np.newaxis]
        )
        dp4 = CoilSet.dot_product_rank3_tensor(numerator, numerator_deriv)
        return (
            numerator_deriv / numerator_norm[:, :, np.newaxis]
            - (dp4 / numerator_norm ** 3)[:, :, np.newaxis] * numerator
        )

    def compute_binormal(tangent, normal):
        """ Computes the binormal vector of the coils, B = T x N """
        return np.cross(tangent, normal)

    def compute_binormal_deriv(tangent, normal, tangent_deriv, normal_deriv):
        return np.cross(tangent_deriv, normal) + np.cross(tangent, normal_deriv)

    def compute_alpha(coil_data, theta, fr):
        NC, NS, _, NFR, _, _, _, _, _, NR = coil_data
        alpha = np.zeros((NC, NS + 1))
        alpha += theta * NR / 2
        Ac = fr[0]
        As = fr[1]
        for m in range(NFR):
            arg = theta * m
            carg = np.cos(arg)
            sarg = np.sin(arg)
            alpha += (
                Ac[:, np.newaxis, m] * carg[np.newaxis, :]
                + As[:, np.newaxis, m] * sarg[np.newaxis, :]
            )
        return alpha

    def compute_alpha_1(coil_data, theta, fr):
        NC, NS, _, NFR, _, _, _, _, _, NR = coil_data
        alpha_1 = np.zeros((NC, NS + 1))
        alpha_1 += NR / 2
        Ac = fr[0]
        As = fr[1]
        for m in range(NFR):
            arg = theta * m
            carg = np.cos(arg)
            sarg = np.sin(arg)
            alpha_1 += (
                -m * Ac[:, np.newaxis, m] * sarg[np.newaxis, :]
                + m * As[:, np.newaxis, m] * carg[np.newaxis, :]
            )
        return alpha_1

    def compute_frame(coil_data, theta, fr, N, B):
        """
		Computes the vectors v1 and v2 for each coil. v1 and v2 are rotated relative to
		the normal and binormal frame by an amount alpha. Alpha is parametrized by a Fourier series.
		"""
        alpha = CoilSet.compute_alpha(coil_data, theta, fr)
        calpha = np.cos(alpha)
        salpha = np.sin(alpha)
        v1 = calpha[:, :, np.newaxis] * N - salpha[:, :, np.newaxis] * B
        v2 = salpha[:, :, np.newaxis] * N + calpha[:, :, np.newaxis] * B
        return v1, v2

    def compute_frame_derivative(coil_data, theta, params, frame, r1, r2, r_centroid):
        _, N, B = frame
        _, fr = params
        alpha = CoilSet.compute_alpha(coil_data, theta, fr)
        calpha = np.cos(alpha)
        salpha = np.sin(alpha)
        alpha1 = CoilSet.compute_alpha_1(coil_data, theta, fr)
        _, dNdt, dBdt = CoilSet.compute_com_deriv(params, frame, r1, r2, r_centroid)
        dv1_dt = (
            calpha[:, :, np.newaxis] * dNdt
            - salpha[:, :, np.newaxis] * dBdt
            - salpha[:, :, np.newaxis] * N * alpha1[:, :, np.newaxis]
            - calpha[:, :, np.newaxis] * B * alpha1[:, :, np.newaxis]
        )
        dv2_dt = (
            salpha[:, :, np.newaxis] * dNdt
            + calpha[:, :, np.newaxis] * dBdt
            + calpha[:, :, np.newaxis] * N * alpha1[:, :, np.newaxis]
            - salpha[:, :, np.newaxis] * B * alpha1[:, :, np.newaxis]
        )
        return dv1_dt, dv2_dt

    def compute_r(coil_data, theta, fr, normal, binormal, r_centroid):
        """
		Computes the position of the multi-filament coils.

		r is a NC x NS + 1 x NNR x NBR x 3 array which holds the coil endpoints
		dl is a NC x NS x NNR x NBR x 3 array which computes the length of the NS segments
		r_middle is a NC x NS x NNR x NBR x 3 array which computes the midpoint of each of the NS segments

		"""
        NC, NS, NF, NFR, ln, lb, NNR, NBR, _, _ = coil_data
        v1, v2 = CoilSet.compute_frame(coil_data, theta, fr, normal, binormal)
        r = np.zeros((NC, NS + 1, NNR, NBR, 3))
        r += r_centroid[:, :, np.newaxis, np.newaxis, :]
        for n in range(NNR):
            for b in range(NBR):
                r = index_add(
                    r,
                    index[:, :, n, b, :],
                    (n - 0.5 * (NNR - 1)) * ln * v1 + (b - 0.5 * (NBR - 1)) * lb * v2,
                )
        return r[:, :-1, :, :, :]

    def compute_dl(coil_data, theta, params, frame, r1, r2, r_centroid):
        NC, NS, NF, NFR, ln, lb, NNR, NBR, _, _ = coil_data
        dl = np.zeros((NC, NS + 1, NNR, NBR, 3))
        dl += r1[:, :, np.newaxis, np.newaxis, :]
        dv1_dt, dv2_dt = CoilSet.compute_frame_derivative(
            coil_data, theta, params, frame, r1, r2, r_centroid
        )
        for n in range(NNR):
            for b in range(NBR):
                dl = index_add(
                    dl,
                    index[:, :, n, b, :],
                    (n - 0.5 * (NNR - 1)) * ln * dv1_dt
                    + (b - 0.5 * (NBR - 1)) * lb * dv2_dt,
                )
        return dl[:, :-1, :, :, :] * (2 * PI / NS)

    def compute_torsion(r1, r2, r3):
        cross12 = np.cross(r1, r2)
        top = (
            cross12[:, :, 0] * r3[:, :, 0]
            + cross12[:, :, 1] * r3[:, :, 1]
            + cross12[:, :, 2] * r3[:, :, 2]
        )
        bottom = np.linalg.norm(cross12, axis=-1) ** 2
        return top / bottom  # NC x NS+1

    def compute_mean_torsion(torsion):
        return np.mean(torsion[:, :-1], axis=-1)

    def writeXYZ(coil_data, params, filename):
        NC, NS, _, _, _, _, NNR, NBR, _, _ = coil_data
        _, _, r, _, _ = CoilSet.get_outputs(coil_data, params)
        with open(filename, "w") as f:
            f.write("periods {}\n".format(0))
            f.write("begin filament\n")
            f.write("FOCUSADD Coils\n")
            for i in range(NC):
                for n in range(NNR):
                    for b in range(NBR):
                        for s in range(NS):
                            f.write(
                                "{} {} {} {}\n".format(
                                    r[i, s, n, b, 0],
                                    r[i, s, n, b, 1],
                                    r[i, s, n, b, 2],
                                    I[i],
                                )
                            )
                        f.write(
                            "{} {} {} {} {} {}\n".format(
                                r[i, 0, n, b, 0],
                                r[i, 0, n, b, 1],
                                r[i, 0, n, b, 2],
                                0.0,
                                "{}{}{}".format(i, n, b),
                                "coil/filament1/filament2",
                            )
                        )
