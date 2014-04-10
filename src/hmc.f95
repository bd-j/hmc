      subroutine leapfrog(ndim,  epsilon, q, p, grad, qprime, pprime, gradprime)
        ! This subroutine makes one leapfrog step in parameter space
        !
        ! Inputs
        ! ------
        !
        ! ndim [integer]:
        !   The dimension of the parameter space.
        !
        ! epsilon [double precision]:
        !   The proposal scale (a tuning parameter). 
        !
        ! q [double precision (ndim)]:
        !   The starting position in parameter space.
        !
        ! p [double precision (ndim)]:
        !   The starting momentum in each parameter.
        !
        ! grad [double precision (ndim)]:
        !   The gradient of the log of the probability in each direction.
        !
        ! Outputs
        ! -------
        !
        ! qprime [double precision (ndim)]:
        !   The new position in parameter space.
        !
        ! pprime [double precision (ndim)]:
        !   The new momentum in parameter space.
        !
        ! lnP [double precision]:
        !   The value of the log-probability function at positions qprime.
        !
        ! accept [integer]:
        !   A binary indicating whether or not each proposal was
        !   accepted.
        implicit none

        integer, intent(in) :: ndim 
        double precision, intent(in) :: epsilon
        double precision, intent(in), dimension(ndim) :: q, p, grad
        double precision, intent(inout), dimension(ndim) :: qprime, pprime, gradprime

        pprime = p + epsilon/2. * grad
        qprime = q + epsilon * pprime
        call model_lnprob_grad(ndim, qprime, gradprime)
        pprime = pprime + epsilon/2 * gradprime        

      end subroutine

      subroutine hmc_advance(ndim, epsilon, length, qin, qprime, lnP, accept)

        ! This subroutine advances an HMC sampler by one trajectory
        !
        ! Inputs
        ! ------
        !
        ! ndim [integer]:
        !   The dimension of the parameter space.
        !
        ! epsilon [double precision]:
        !   The proposal scale (a tuning parameter). 
        !
        ! length [integer]:
        !    The number of leapfrog steps to take
        !
        ! qin [double precision (ndim)]:
        !   The starting position in parameter space.
        !
        !
        ! Outputs
        ! -------
        !
        ! qprime [double precision (ndim)]:
        !   The final positions in parameter space.
        !
        ! lnP [double precision]:
        !   The value of the log-probability function at positions qprime.
        !
        ! accept [integer]:
        !   A binary indicating whether or not each proposal was
        !   accepted.

        implicit none

        integer, intent(in) :: ndim, length
        double precision, intent(in) :: epsilon
        double precision, intent(in), dimension(ndim) :: qin

        double precision, intent(inout), dimension(ndim) :: qprime
        double precision, intent(inout) :: lnP
        integer, intent(inout) :: accept

        double precision, dimension(ndim) :: p, pin, q, pprime, gradprime, grad
        double precision :: lnP0, r, alpha, dU, dK
        integer :: i

        !initialize random momenta.  This needs to be a gaussian given the kinetic energy formulation we use
        do i = 1, ndim
           call random_normal(r)
           
           pin(i) = r
        end do
        
        !leap 'length' steps
        q = qin
        p = pin

        !find the first gradient and lnp
        call model_lnprob_grad(q, grad)
        call model_lnprob(q, lnP0)

        do i = 1, length
           call leapfrog(ndim, epsilon, q, p, grad, qprime, pprime, gradprime)
           q = qprime
           p = pprime
           !write(*,*) q(:)
           grad = gradprime
        end do
        !probability at the proposed location
        call model_lnprob(ndim, q, lnP)
        dU =  lnP0 - lnP !change in potential = negative change in lnP
        dK = 0.0 !change in kinetic
        do i = 1, ndim
           dK = dK + 0.5 * (p(i) * p(i)  - pin(i) * pin(i)) 
        end do
        alpha = exp(-dU - dK)
        call random_number(r)
        accept = 1
        if (r .gt. alpha) then !reject and reset
           qprime = qin
           lnP = lnP0   
           accept = 0
        end if

      end subroutine

      subroutine random_normal(k)

        implicit none
        double precision, intent(inout) :: k
        double precision :: s, u, v

        do 
           call random_number(u)
           call random_number(v)
           u = u*2.0d0 - 1.0d0
           s = u * u + v * v
           if ((s < 1).and.(s > 0)) exit
        end do
        k =  u * sqrt(-2 * log(s) / s)
      end subroutine

      ! See: http://gcc.gnu.org/onlinedocs/gfortran/RANDOM_005fSEED.html
      subroutine init_random_seed ()

        implicit none

        integer, allocatable :: seed(:)
        integer :: i, n, un, istat, dt(8), pid, t(2), s
        integer(8) :: count, tms

        call random_seed(size = n)
        allocate(seed(n))
        ! First try if the OS provides a random number generator
        open(newunit=un, file="/dev/urandom", access="stream", &
          form="unformatted", action="read", status="old", iostat=istat)
        if (istat == 0) then
          read(un) seed
          close(un)
        else
          ! Fallback to XOR:ing the current time and pid. The PID is
          ! useful in case one launches multiple instances of the same
          ! program in parallel.
          call system_clock(count)
          if (count /= 0) then
            t = transfer(count, t)
          else
            call date_and_time(values=dt)
            tms = (dt(1) - 1970) * 365_8 * 24 * 60 * 60 * 1000 &
              + dt(2) * 31_8 * 24 * 60 * 60 * 1000 &
              + dt(3) * 24 * 60 * 60 * 60 * 1000 &
              + dt(5) * 60 * 60 * 1000 &
              + dt(6) * 60 * 1000 + dt(7) * 1000 &
              + dt(8)
            t = transfer(tms, t)
          end if
          s = ieor(t(1), t(2))
          pid = getpid() + 1099279 ! Add a prime
          s = ieor(s, pid)
          if (n >= 3) then
            seed(1) = t(1) + 36269
            seed(2) = t(2) + 72551
            seed(3) = pid
            if (n > 3) then
              seed(4:) = s + 37 * (/ (i, i = 0, n - 4) /)
            end if
          else
            seed = s + 37 * (/ (i, i = 0, n - 1 ) /)
          end if
        end if
        call random_seed(put=seed)
      end subroutine init_random_seed
