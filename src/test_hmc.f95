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
        double precision, intent(inout) :: lnP

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
        ! q [double precision (ndim)]:
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
        double precision, intent(inout), dimension(ndim) :: grad

        integer :: i
        
        do i=1,ndim
          grad(i) = -q(i)
        enddo
        !well, that was easy
      end subroutine

      program main

        implicit none

        integer, parameter ::length =10, ndim=2
        double precision, dimension(ndim) :: pos
        double precision :: lp, r, epsilon = 0.5
        integer :: len
        !integer, dimension(:) :: accept

        integer :: i, j, acc

        ! First seed the random number generator... don't forget this!
        call init_random_seed ()

        ! Loop over the number of dimensions and initialize each one
        ! in the range `(0.5, 0.5)`.
        do i=1,ndim
           call random_number(pos(i))
           pos(i) = pos(i) - 0.5d0
        enddo


        ! Run a production chain  of 200 iterations or trajectories.
        do i=1,10000
          !choose a random length up to 'length'
          call random_number(r)
          len = int(r * length)
          call hmc_advance (ndim,epsilon,len,pos,pos,lp,acc)
          write(*,*) pos(:), lp, acc
        enddo

      end program
