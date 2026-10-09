import numpy as np
from types import SimpleNamespace
from revosim.wrench import read_wrench


class T:
    def __init__(self, a):
        self.a = np.asarray(a)

    def __getitem__(self, index):
        return T(self.a[index])

    def __len__(self):
        return len(self.a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class Contacts:
    def get_contact_data(self, dt):
        return list(
            map(
                T,
                [
                    np.array([[2.0], [3.0], [0.0], [0.0]]),
                    [[1, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0]],
                    [[0, 0, 1], [0, 0, 1], [0, 0, 0], [0, 0, 0]],
                    [0] * 4,
                    [[1], [1]],
                    [[0], [1]],
                ],
            )
        )

    def get_contact_force_matrix(self, dt):
        return T([[[0, 0, 2]], [[0, 0, 3]]])

    def get_friction_data(self, dt):
        return list(
            map(
                T,
                [
                    [[1, 0, 0], [0, 2, 0], [0, 0, 0], [0, 0, 0]],
                    [[0, 1, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0]],
                    [[1], [1]],
                    [[0], [1]],
                ],
            )
        )


def test_contact_wrench_sums_selected_bodies_and_both_contact_components():
    w, contact = read_wrench(
        SimpleNamespace(contact_physx_view=Contacts()), 1 / 240, np.zeros(3), np.eye(3)
    )
    np.testing.assert_allclose(w, [1, 2, 5, 3, -2, 1])
    assert contact


class BalancedContacts(Contacts):
    def get_contact_data(self, dt):
        return list(
            map(
                T,
                [
                    [[2.0], [2.0], [0.0], [0.0]],
                    [[1, 0, 0], [-1, 0, 0], [0, 0, 0], [0, 0, 0]],
                    [[0, 0, 1], [0, 0, -1], [0, 0, 0], [0, 0, 0]],
                    [0] * 4,
                    [[2]],
                    [[0]],
                ],
            )
        )

    def get_contact_force_matrix(self, dt):
        return T([[[0, 0, 0]]])

    def get_friction_data(self, dt):
        return list(map(T, [np.zeros((4, 3)), np.zeros((4, 3)), [[0]], [[0]]]))


def test_cancelling_forces_preserve_nonzero_moment():
    w, contact = read_wrench(
        SimpleNamespace(contact_physx_view=BalancedContacts()),
        1 / 240,
        np.zeros(3),
        np.eye(3),
    )
    np.testing.assert_allclose(w, [0, 0, 0, 0, -4, 0])
    assert contact


class EmptyContacts(BalancedContacts):
    def get_contact_data(self, dt):
        return list(
            map(
                T,
                [
                    np.zeros((4, 1)),
                    np.zeros((4, 3)),
                    np.zeros((4, 3)),
                    [0] * 4,
                    [[0]],
                    [[0]],
                ],
            )
        )


def test_empty_buffers_have_no_contact_or_wrench():
    w, contact = read_wrench(
        SimpleNamespace(contact_physx_view=EmptyContacts()),
        1 / 240,
        np.zeros(3),
        np.eye(3),
    )
    np.testing.assert_array_equal(w, np.zeros(6))
    assert not contact


class MultiObjectContacts(Contacts):
    def get_contact_data(self, dt):
        data = super().get_contact_data(dt)
        data[-2:] = [T([[1, 1]]), T([[0, 1]])]
        return data

    def get_contact_force_matrix(self, dt):
        return T([[[0, 0, 2], [0, 0, 3]]])

    def get_friction_data(self, dt):
        data = super().get_friction_data(dt)
        data[-2:] = [T([[1, 1]]), T([[0, 1]])]
        return data


def test_tip_wrench_includes_every_object_filter():
    w, contact = read_wrench(
        SimpleNamespace(contact_physx_view=MultiObjectContacts()),
        1 / 240,
        np.zeros(3),
        np.eye(3),
    )
    np.testing.assert_allclose(w, [1, 2, 5, 3, -2, 1])
    assert contact
