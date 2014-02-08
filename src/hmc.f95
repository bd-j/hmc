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
        double precision, intent(out), dimension(ndim) :: qprime, pprime, gradprime

        pprime = p + epsilon/2. * grad
        qprime = q + epsilon * pprime
        call model_lnprob_grad(ndim, qprime, gradprime)
        pprime = pprime + epsilon/2 * gradprime        

      end subroutine

      subroutine hmc_advance(ndim, epsilon, length, q, qprime, lnP, accept)

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
        ! q [double precision (ndim)]:
        !   The starting position in parameter space.
        !
        ! length [integer]:
        !    The number of leapfrog steps to take
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
        double precision, intent(in), dimension(ndim) :: q

        double precision, intent(out), dimension(ndim) :: qprime
        double precision, intent(out) :: lnP
        integer, intent(out) :: accept

        double precision, dimension(ndim) :: p, pin, pprime, qin, gradprime, grad
        double precision :: lnP0, r, alpha, dU, dK
        integer :: j, k, l

        !initialize random momenta
        do j = 1, ndim
           call random_number(r)
           p(j) = r
        end do
        !find the first gradient and lnp
        call model_lnprob_grad(q, grad)
        call model_lnprob(q, lnP0)
        
        !leap 'length' steps
        qin = q
        pin = p
        do k = 1, length
           call leapfrog(ndim, epsilon, qin, p, grad, qprime, pprime, gradprime)
           qin = qprime
           p = pprime
           grad = gradprime
        end do
        !probability at the proposed location
        call model_lnprob(ndim, q, lnP)
        dU =  lnP0 - lnP !change in potential = negative change in lnP
        dK = 0.0 !change in kinetic
        do l = 1, ndim
           dK = dK + 0.5 * (p(l) * p(l)  - pin(l) * pin(l)) 
        end do
        alpha = exp(-dU - dK)
        call random_number(r)
        accept = 1
        if (r .gt. alpha) then !reject
           qprime = q
           lnP = lnP0   
           accept = 0
        end if

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
