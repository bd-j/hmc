      subroutine model_lnprob (ndim, q, lnP)

        ! This subroutine demonstrates the syntax for implementing a
        ! log-probability function. This particular example implements
        ! an isotropic `ndim`-dimensional Gaussian with unit variance.
        !
        ! Inputs
        ! ------
        !
        ! ndim [integer]:
        !   The dimension of the parameter space.
        !
        ! q [double precision (ndim)]:
        !   The position in parameter space where the probability should
        !   be computed.
        !
        ! Outputs
        ! -------
        !
        ! lnP [double precision]:
        !   The log-probability computed at q.

        implicit none

        integer, intent(in) :: ndim
        double precision, intent(in), dimension(ndim) :: q
        double precision, intent(out) :: lnP

        integer :: i

        lnP = 1.d0
        do i=1,ndim
          lnP = lnP + q(i)*q(i)
        enddo
        lnP = -0.5d0 * lnP

      end subroutine


      subroutine model_lnprob_grad (ndim, q, grad)

        ! This subroutine demonstrates the syntax for implementing a
        ! log-probability function. This particular example implements
        ! an isotropic `ndim`-dimensional Gaussian with unit variance.
        !
        ! Inputs
        ! ------
        !
        ! ndim [integer]:
        !   The dimension of the parameter space.
        !
        ! 1 [double precision (ndim)]:
        !   The position in parameter space where the probability should
        !   be computed.
        !
        ! Outputs
        ! -------
        !
        ! grad [double precision (ndim)]:
        !   The gradient of log-probability computed at q.

        implicit none

        integer, intent(in) :: ndim
        double precision, intent(in), dimension(ndim) :: q
        double precision, intent(out), dimension(ndim) :: grad

        integer :: i
        
        do i=1,ndim
          grad(i) = -q(i)
        enddo
        !well, that was easy
      end subroutine

      program main

        implicit none

        integer, parameter :: length=20, ndim=2
        double precision, dimension(ndim) :: pos
        double precision :: lp, epsilon = 0.2
        integer, dimension(length) :: accept

        integer :: i, j

        ! First seed the random number generator... don't forget this!
        call init_random_seed ()

        ! Loop over the number of dimensions and initialize each one
        ! in the range `(0.5, 0.5)`.
        do i=1,ndim
           call random_number(pos(i))
           pos(i) = pos(i) - 0.5d0
        enddo


        ! Run a production chain  of 200 iterations or trajectories.
        do i=1,200
          ! You'll notice that I'm overwriting the position and
          ! log-probability of the ensemble at each step. This works but
          ! you also have the option of saving the samples by giving
          ! different input and output arguments.
          call hmc_advance (ndim,epsilon,length,pos,pos,lp,accept)
          write(*,*) pos(:), lp
        enddo

      end program
